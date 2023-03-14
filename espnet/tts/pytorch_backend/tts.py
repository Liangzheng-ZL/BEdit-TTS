#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""E2E-TTS training / decoding functions."""

import copy
import json
import logging
import math
import os
import time

import chainer
import kaldiio
import matplotlib
import numpy as np
import torch
from chainer import training
from chainer.training import extensions
from tensorboardX import SummaryWriter

from espnet.asr.asr_utils import (get_model_conf, snapshot_object, torch_load,
                                  torch_resume, torch_save, torch_snapshot)
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop, set_early_stop

matplotlib.use('Agg')


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator."""

    def __init__(self, model, iterator, target, device):
        """Initilize module.

        Args:
            model (torch.nn.Module): Pytorch model instance.
            iterator (chainer.dataset.Iterator): Iterator for validation.
            target (chainer.Chain): Dummy chain instance.
            device (torch.device): The device to be used in evaluation.

        """
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.device = device

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        """Evaluate over validation iterator."""
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                if isinstance(batch, tuple):
                    x = tuple(arr.to(self.device) for arr in batch)
                else:
                    x = batch
                    for key in x.keys():
                        x[key] = x[key].to(self.device)
                observation = {}
                with chainer.reporter.report_scope(observation):
                    # convert to torch tensor
                    if isinstance(x, tuple):
                        self.model(*x)
                    else:
                        self.model(**x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    """Custom updater."""

    def __init__(self, model, grad_clip, iterator, optimizer, device, accum_grad=1):
        """Initilize module.

        Args:
            model (torch.nn.Module) model: Pytorch model instance.
            grad_clip (float) grad_clip : The gradient clipping value.
            iterator (chainer.dataset.Iterator): Iterator for training.
            optimizer (torch.optim.Optimizer) : Pytorch optimizer instance.
            device (torch.device): The device to be used in training.

        """
        super(CustomUpdater, self).__init__(iterator, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.device = device
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.accum_grad = accum_grad
        self.forward_count = 0

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Update model one step."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        if isinstance(batch, tuple):
            x = tuple(arr.to(self.device) for arr in batch)
        else:
            x = batch
            for key in x.keys():
                x[key] = x[key].to(self.device)

        # compute loss and gradient
        if isinstance(x, tuple):
            loss = self.model(*x).mean() / self.accum_grad
        else:
            loss = self.model(**x).mean() / self.accum_grad
        loss.backward()

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0

        # compute the gradient norm to check if it is normal or not
        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip)
        logging.debug('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        optimizer.zero_grad()

    def update(self):
        """Run update function."""
        self.update_core()
        if self.forward_count == 0:
            self.iteration += 1


class CustomConverter(object):
    """Custom converter."""

    def __init__(self):
        """Initilize module."""
        # NOTE: keep as class for future development
        pass

    def __call__(self, batch, device=torch.device('cpu')):
        """Convert a given batch.

        Args:
            batch (list): List of ndarrays.
            device (torch.device): The device to be send.

        Returns:
            dict: Dict of converted tensors.

        Examples:
            >>> batch = [([np.arange(5), np.arange(3)],
                          [np.random.randn(8, 2), np.random.randn(4, 2)],
                          None, None)]
            >>> conveter = CustomConverter()
            >>> conveter(batch, torch.device("cpu"))
            {'xs': tensor([[0, 1, 2, 3, 4],
                           [0, 1, 2, 0, 0]]),
             'ilens': tensor([5, 3]),
             'ys': tensor([[[-0.4197, -1.1157],
                            [-1.5837, -0.4299],
                            [-2.0491,  0.9215],
                            [-2.4326,  0.8891],
                            [ 1.2323,  1.7388],
                            [-0.3228,  0.6656],
                            [-0.6025,  1.3693],
                            [-1.0778,  1.3447]],
                           [[ 0.1768, -0.3119],
                            [ 0.4386,  2.5354],
                            [-1.2181, -0.5918],
                            [-0.6858, -0.8843],
                            [ 0.0000,  0.0000],
                            [ 0.0000,  0.0000],
                            [ 0.0000,  0.0000],
                            [ 0.0000,  0.0000]]]),
             'labels': tensor([[0., 0., 0., 0., 0., 0., 0., 1.],
                               [0., 0., 0., 1., 1., 1., 1., 1.]]),
             'olens': tensor([8, 4])}

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, spkids, pos, phn_pos, ys, durs, variances = batch[0]

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)

        mask_ilens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)
        mask_mel_inputs = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)

        # make labels for stop prediction
        labels = ys.new_zeros(ys.size(0), ys.size(1))
        for i, l in enumerate(olens):
            labels[i, l - 1:] = 1.0

        # print("make batch xs shape ", xs.shape)

        # prepare dict
        new_batch = {
            "xs": xs,
            "ilens": ilens,
            "ys": ys,
            "labels": labels,
            "olens": olens,
            "mask_mel_inputs": mask_mel_inputs,
            "mask_ilens": mask_ilens,
        }

        durs = pad_list([torch.from_numpy(dur).long() for dur in durs], 0).to(device)
        new_batch['ds'] = durs
        new_batch['spkids'] = torch.LongTensor(spkids).to(device)
        new_batch['pos'] = pad_list([torch.from_numpy(po).long() for po in pos], 0).to(device)
        new_batch['phn_pos'] = pad_list([torch.from_numpy(ppo).long() for ppo in phn_pos], 0).to(device)
        variances = pad_list([torch.from_numpy(variance).float() for variance in variances], 0).to(device)
        new_batch['ps'] = variances[:, :, :4]
        new_batch['es'] = variances[:, :, 4:]
        new_batch['pes'] = variances

        return new_batch


def train(args):
    """Train E2E-TTS model."""
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())

    # reverse input and output dimension
    idim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, odim, args)
    assert isinstance(model, TTSInterface)
    logging.info(model)
    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import \
            get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, 'target', reporter)
    setattr(optimizer, 'serialize', lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    if use_sortagrad:
        args.batch_sort_key = "input"
    # make minibatch list (variable length)
    train_batchset = make_batchset(train_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out, args.minibatches,
                                   batch_sort_key=args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                                   shortest_first=use_sortagrad,
                                   count=args.batch_count,
                                   batch_bins=args.batch_bins,
                                   batch_frames_in=args.batch_frames_in,
                                   batch_frames_out=args.batch_frames_out,
                                   batch_frames_inout=args.batch_frames_inout,
                                   swap_io=True, iaxis=0, oaxis=0)
    valid_batchset = make_batchset(valid_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out, args.minibatches,
                                   batch_sort_key=args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                                   count=args.batch_count,
                                   batch_bins=args.batch_bins,
                                   batch_frames_in=args.batch_frames_in,
                                   batch_frames_out=args.batch_frames_out,
                                   batch_frames_inout=args.batch_frames_inout,
                                   swap_io=True, iaxis=0, oaxis=0)

    load_tr = LoadInputsAndTargets(
        mode='tts',
        use_speaker_embedding=args.use_speaker_embedding,
        use_second_target=args.use_second_target,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True},  # Switch the mode of preprocessing
        keep_all_data_on_mem=args.keep_all_data_on_mem,
    )

    load_cv = LoadInputsAndTargets(
        mode='tts',
        use_speaker_embedding=args.use_speaker_embedding,
        use_second_target=args.use_second_target,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False},  # Switch the mode of preprocessing
        keep_all_data_on_mem=args.keep_all_data_on_mem,
    )

    converter = CustomConverter()
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    train_iter = {'main': ChainerDataLoader(
        dataset=TransformDataset(train_batchset, lambda data: converter([load_tr(data)])),
        batch_size=1, num_workers=args.num_iter_processes,
        shuffle=not use_sortagrad, collate_fn=lambda x: x[0])}
    valid_iter = {'main': ChainerDataLoader(
        dataset=TransformDataset(valid_batchset, lambda data: converter([load_cv(data)])),
        batch_size=1, shuffle=False, collate_fn=lambda x: x[0],
        num_workers=args.num_iter_processes)}

    # Set up a trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer, device, args.accum_grad)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # set intervals
    eval_interval = (args.eval_interval_epochs, 'epoch')
    save_interval = (args.save_interval_epochs, 'epoch')
    report_interval = (args.report_interval_iters, 'iteration')

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(
        model, valid_iter, reporter, device), trigger=eval_interval)

    # Save snapshot for each epoch
    trainer.extend(torch_snapshot(), trigger=save_interval)

    # Save best models
    trainer.extend(snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger(
                       'validation/main/loss', trigger=eval_interval))

    # Save attention figure for each epoch
    # args.num_save_attention = 0
    if args.num_save_attention > 0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn, data, args.outdir + '/att_ws',
            converter=converter,
            transform=load_cv,
            device=device, reverse=True)
        trainer.extend(att_reporter, trigger=eval_interval)
    else:
        att_reporter = None

    # Make a plot for training and validation values
    if hasattr(model, "module"):
        base_plot_keys = model.module.base_plot_keys
    else:
        base_plot_keys = model.base_plot_keys
    plot_keys = []
    for key in base_plot_keys:
        plot_key = ['main/' + key, 'validation/main/' + key]
        trainer.extend(extensions.PlotReport(
            plot_key, 'epoch', file_name=key + '.png'), trigger=eval_interval)
        plot_keys += plot_key
    trainer.extend(extensions.PlotReport(
        plot_keys, 'epoch', file_name='all_loss.png'), trigger=eval_interval)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=report_interval))
    report_keys = ['epoch', 'iteration', 'elapsed_time'] + plot_keys
    trainer.extend(extensions.PrintReport(report_keys), trigger=report_interval)
    trainer.extend(extensions.ProgressBar(), trigger=report_interval)

    set_early_stop(trainer, args)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer, att_reporter), trigger=report_interval)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def prepare(args):
    """Decode with E2E-TTS model."""
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info('args: ' + key + ': ' + str(vars(args)[key]))

    # define model
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, TTSInterface)
    logging.info(model)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    torch_load(args.model, model)

    return train_args, model

def decode(args):
    # prepare
    train_args, model = prepare(args)

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)
    model.eval()
    
    # log the number of params of model
    total_num_params = sum(p.numel() for p in model.parameters())
    total_train_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"total number of parameters: {total_num_params}, training params: {total_train_num_params}")

    # read json data
    with open(args.json, 'rb') as f:
        js = json.load(f)['utts']

    # check directory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    load_output = args.use_gt_condition
    load_inputs_and_targets = LoadInputsAndTargets(
        mode='tts', load_input=load_output, sort_in_input_length=False,
        use_speaker_embedding=train_args.use_speaker_embedding,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        is_training=False,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    # define function for plot prob and att_ws
    def _plot_and_save(array, figname, figsize=(6, 4), dpi=150):
        import matplotlib.pyplot as plt
        shape = array.shape
        if len(shape) == 1:
            # for eos probability
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(array)
            plt.xlabel("Frame")
            plt.ylabel("Probability")
            plt.ylim([0, 1])
        elif len(shape) == 2:
            # for tacotron 2 attention weights, whose shape is (out_length, in_length)
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(array, aspect="auto")
            plt.xlabel("Input")
            plt.ylabel("Output")
        elif len(shape) == 4:
            # for transformer attention weights, whose shape is (#leyers, #heads, out_length, in_length)
            plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
            for idx1, xs in enumerate(array):
                for idx2, x in enumerate(xs, 1):
                    plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                    plt.imshow(x, aspect="auto")
                    plt.xlabel("Input")
                    plt.ylabel("Output")
        else:
            raise NotImplementedError("Support only from 1D to 4D array.")
        plt.tight_layout()
        if not os.path.exists(os.path.dirname(figname)):
            # NOTE: exist_ok = True is needed for parallel process decoding
            os.makedirs(os.path.dirname(figname), exist_ok=True)
        plt.savefig(figname)
        plt.close()


    with torch.no_grad(), \
         kaldiio.WriteHelper('ark,scp:{o}.ark,{o}.scp'.format(o=args.out)) as f:
        # with torch.no_grad(), \
        #  kaldiio.WriteHelper('ark,scp:{o_pe}.ark,{o_pe}.scp'.format(o_pe=args.out+'.pe')) as f_pe:
            # fr_dur = open(args.out+'.dur', 'w')
            dur_json = []
            for idx, utt_id in enumerate(js.keys()):
                batch = [(utt_id, js[utt_id])]
                data = load_inputs_and_targets(batch)

                xs = torch.LongTensor(data[0]).to(device)

                spkids = torch.LongTensor(data[1]).to(device)

                # position
                pos = torch.IntTensor(data[2]).to(device)
                phn_pos = torch.IntTensor(data[3]).to(device)

                # ys = ds = es = ps = None
                # if load_output:
                ys = torch.FloatTensor(data[4]).to(device)
                ds = torch.LongTensor(data[5]).to(device)
                vars = torch.FloatTensor(data[6]).to(device)
                ps = vars[:, :, :4]
                es = vars[:, :, 4:]

                mask_mel_inputs = torch.FloatTensor(data[4]).to(device)

                # decode and write
                start_time = time.time()
                outs, probs, att_ws, pe_outs, dur_outs, new_phn_position, new_position \
                    = model.inference(xs, spkids, 
                    args.is_extra_spk, args.use_gt_condition,
                    pos=pos, phn_pos=phn_pos, ys=ys, ds=ds, es=es, ps=ps, pes=vars,
                    mask_mel_inputs=mask_mel_inputs)
    
                logging.info("inference speed = %s msec / frame." % (
                        (time.time() - start_time) / (int(outs.size(0)) * 1000)))
                if outs.size(0) == xs.size(1) * args.maxlenratio:
                    logging.warning("output length reaches maximum length (%s)." % utt_id)
                logging.info('(%d/%d) %s (size:%d->%d)' % (
                    idx + 1, len(js.keys()), utt_id, xs.size(1), outs.size(0)))
                f[utt_id] = outs.cpu().numpy()
                # f_pe[utt_id] = pe_outs.cpu().numpy()
                dur_item = {}
                dur_item["utt_id"] = utt_id
                dur_item["phn_position"] = [x.item() for x in new_phn_position]
                dur_item["position"] = [x.item() for x in new_position]
                dur_item["duration"] = dur_outs.cpu().numpy().tolist()
                dur_sum = 0
                for dur_phn in dur_item["duration"]:
                    dur_sum+=dur_phn
                dur_item["frames"] = dur_sum
                dur_json.append(dur_item)
            tmp = json.dumps(dur_json, indent=4)
            with open(args.out+'.dur.json', 'w') as f_dur:
                f_dur.write(tmp)
                f_dur.close
                # fr_dur.write(utt_id+' '+(dur_outs.cpu().numpy())+'\n')

                # plot prob and att_ws
                if probs is not None:
                    _plot_and_save(probs.cpu().numpy(), os.path.dirname(args.out) + "/probs/%s_prob.png" % utt_id)
                if att_ws is not None:
                    _plot_and_save(att_ws.cpu().numpy(), os.path.dirname(args.out) + "/att_ws/%s_att_ws.png" % utt_id)

def transfer(args):
    # prepare
    train_args, model = prepare(args)

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)
    model.eval()

    # read json data
    with open(args.json, 'rb') as f:
        js = json.load(f)['utts']

    # check directory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='tts', load_input=True, sort_in_input_length=False,
        use_speaker_embedding=train_args.use_speaker_embedding,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    with torch.no_grad(), \
         kaldiio.WriteHelper('ark,scp:{o}.ark,{o}.scp'.format(o=args.out)) as f:

        for idx, utt_id in enumerate(js.keys()):
            batch = [(utt_id, js[utt_id])]
            data = load_inputs_and_targets(batch)

            xs = torch.LongTensor(data[0]).to(device)
            src_spkids = torch.LongTensor(data[1]).to(device)
            tgt_spkids = torch.zeros_like(src_spkids) + args.tgt_spkid
            ys = torch.FloatTensor(data[2]).to(device)
            ds = torch.LongTensor(data[3]).to(device)

            # decode and write
            start_time = time.time()

            outs, probs, att_ws = model.transfer(xs, src_spkids, tgt_spkids, is_extra_src_spk=args.is_extra_src_spk, is_extra_tgt_spk=args.is_extra_tgt_spk, ys=ys, ds=ds)

            logging.info("inference speed = %s msec / frame." % (
                    (time.time() - start_time) / (int(outs.size(0)) * 1000)))
            if outs.size(0) == xs.size(1) * args.maxlenratio:
                logging.warning("output length reaches maximum length (%s)." % utt_id)
            logging.info('(%d/%d) %s (size:%d->%d)' % (
                idx + 1, len(js.keys()), utt_id, xs.size(1), outs.size(0)))
            f[utt_id] = outs.cpu().numpy()

def finetune(args):
    # prepare
    train_args, model = prepare(args)

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        if train_args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                train_args.batch_size, train_args.batch_size * args.ngpu))
            train_args.batch_size *= args.ngpu

    # read json data
    with open(args.json, 'rb') as f:
        train_json = json.load(f)['utts']

    use_sortagrad = train_args.sortagrad == -1 or train_args.sortagrad > 0
    if use_sortagrad:
        train_args.batch_sort_key = "input"

    # make minibatch list (variable length)
    train_batchset = make_batchset(train_json, train_args.batch_size,
                                   train_args.maxlen_in, train_args.maxlen_out, train_args.minibatches,
                                   batch_sort_key=train_args.batch_sort_key,
                                   min_batch_size=train_args.ngpu if train_args.ngpu > 1 else 1,
                                   shortest_first=use_sortagrad,
                                   count=train_args.batch_count,
                                   batch_bins=train_args.batch_bins,
                                   batch_frames_in=train_args.batch_frames_in,
                                   batch_frames_out=train_args.batch_frames_out,
                                   batch_frames_inout=train_args.batch_frames_inout,
                                   swap_io=True, iaxis=0, oaxis=0)

    load_tr = LoadInputsAndTargets(
        mode='tts',
        use_speaker_embedding=train_args.use_speaker_embedding,
        use_second_target=train_args.use_second_target,
        preprocess_conf=train_args.preprocess_conf,
        preprocess_args={'train': False},  # Switch the mode of preprocessing
        keep_all_data_on_mem=train_args.keep_all_data_on_mem,
    )

    converter = CustomConverter()
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    train_iter = ChainerDataLoader(
        dataset=TransformDataset(train_batchset, lambda data: converter([load_tr(data)])),
        batch_size=1, num_workers=train_args.num_iter_processes,
        shuffle=not use_sortagrad, collate_fn=lambda x: x[0])

    if hasattr(train_iter, 'reset'):
        train_iter.reset()
        it = train_iter
    else:
        it = copy.copy(train_iter)

    if args.is_extra_spk:
        extra_module_name = 'extra_spk_embed_table'
    else:
        extra_module_name = 'spk_embed_table'

    if isinstance(model, torch.nn.DataParallel):
        extra_module = getattr(model.module, extra_module_name)
    else:
        extra_module = getattr(model, extra_module_name)
    optimizer = torch.optim.Adam(extra_module.parameters(), train_args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        for batch in it:
            if isinstance(batch, tuple):
                x = tuple(arr.to(device) for arr in batch)
            else:
                x = batch
                for key in x.keys():
                    x[key] = x[key].to(device)

            # compute loss and gradient
            if isinstance(x, tuple):
                loss = model(*x, is_extra_spk=args.is_extra_spk).mean()
            else:
                loss = model(**x, is_extra_spk=args.is_extra_spk).mean()
            total_loss += loss.item()
            loss.backward()

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
            logging.debug('grad norm={}'.format(grad_norm))
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.step()
            extra_module.zero_grad()

        logging.info('Epoch %03d, Loss: %.2f' % (epoch + 1, total_loss))

    torch_save(args.out, model)
