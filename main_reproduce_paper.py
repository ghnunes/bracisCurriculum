#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:53:30 2019

@author: guy.hacohen
"""

from argparse import Namespace
from main_train_networks import run_expriment


def curriculum_small_mammals(repeats, output_path="output/"):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

    
def vanilla_small_mammals(repeats, output_path="output/"):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.035,
                     lr_decay_rate=1.8,
                     minimal_lr=1e-4,
                     lr_batch_size=600,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def anti_curriculum_small_mammals(repeats, output_path="output/"):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=200,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def random_small_mammals(repeats, output_path="output/"):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.3,
                     minimal_lr=1e-4,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def case_1_kDN_random16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)

def case_1_kDN_random17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)

def case_1_kDN_random19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)

def case_1_kDN_random14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)

def case_1_DCP_random16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                        model='stVGG',
                        output_path=output_path,
                        verbose=False,
                        optimizer="sgd",
                        curriculum="random",
                        batch_size=100,
                        num_epochs=140,
                        learning_rate=0.03,
                        lr_decay_rate=1.1,
                        minimal_lr=1e-4,
                        lr_batch_size=100,
                        batch_increase=100,
                        increase_amount=1.9,
                        starting_percent=0.04,
                        order="DCP",
                        transfer_net="inception",
                        test_each=50,
                        repeats=repeats,
                        balance=True)
    run_expriment(args)

def case_1_DCP_random17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                        model='stVGG',
                        output_path=output_path,
                        verbose=False,
                        optimizer="sgd",
                        curriculum="random",
                        batch_size=100,
                        num_epochs=140,
                        learning_rate=0.03,
                        lr_decay_rate=1.1,
                        minimal_lr=1e-4,
                        lr_batch_size=100,
                        batch_increase=100,
                        increase_amount=1.9,
                        starting_percent=0.04,
                        order="DCP",
                        transfer_net="inception",
                        test_each=50,
                        repeats=repeats,
                        balance=True)
    run_expriment(args)

def case_1_DCP_random17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                        model='stVGG',
                        output_path=output_path,
                        verbose=False,
                        optimizer="sgd",
                        curriculum="random",
                        batch_size=100,
                        num_epochs=140,
                        learning_rate=0.03,
                        lr_decay_rate=1.1,
                        minimal_lr=1e-4,
                        lr_batch_size=100,
                        batch_increase=100,
                        increase_amount=1.9,
                        starting_percent=0.04,
                        order="DCP",
                        transfer_net="inception",
                        test_each=50,
                        repeats=repeats,
                        balance=True)
    run_expriment(args)

def case_1_DCP_random19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                        model='stVGG',
                        output_path=output_path,
                        verbose=False,
                        optimizer="sgd",
                        curriculum="random",
                        batch_size=100,
                        num_epochs=140,
                        learning_rate=0.03,
                        lr_decay_rate=1.1,
                        minimal_lr=1e-4,
                        lr_batch_size=100,
                        batch_increase=100,
                        increase_amount=1.9,
                        starting_percent=0.04,
                        order="DCP",
                        transfer_net="inception",
                        test_each=50,
                        repeats=repeats,
                        balance=True)
    run_expriment(args)

def case_1_DCP_random14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                        model='stVGG',
                        output_path=output_path,
                        verbose=False,
                        optimizer="sgd",
                        curriculum="random",
                        batch_size=100,
                        num_epochs=140,
                        learning_rate=0.03,
                        lr_decay_rate=1.1,
                        minimal_lr=1e-4,
                        lr_batch_size=100,
                        batch_increase=100,
                        increase_amount=1.9,
                        starting_percent=0.04,
                        order="DCP",
                        transfer_net="inception",
                        test_each=50,
                        repeats=repeats,
                        balance=True)
    run_expriment(args)


def case_1_MV(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="MV",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_CB(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CB",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_N1(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="N1",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_N2(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="N2",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_F1(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="F1",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_F2(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="F2",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_F3(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="F3",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_F4(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="F4",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_LSC(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="LSC",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_LSR(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="LSR",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_Harmfulness(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="Harmfulness",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_Usefulness(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="Usefulness",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

def case_1_cld_curriculum16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_curriculum17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_curriculum19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_curriculum14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_anti16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_anti17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_random16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_random17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_random19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_random14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_curriculum16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_curriculum17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_curriculum19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_curriculum14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_DCP_curriculum16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_DCP_curriculum17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_DCP_curriculum19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_DCP_curriculum14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_anti16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_anti17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_anti19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_kDN_anti14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="kDN",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)


def case_1_DCP_anti16(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_DCP_anti17(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_17",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_DCP_anti19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_DCP_anti14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="DCP",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)



def case_1_cld_anti19(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_19",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def case_1_cld_anti14(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100_subset_14",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="CLD",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)

    run_expriment(args)

def vanilla_cifar10_st_vgg(repeats, output_path="output/"):
    args = Namespace(dataset="cifar10",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)


def curriculum_cifar10_st_vgg(repeats, output_path="output/"):
    args = Namespace(dataset="cifar10",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def vanilla_cifar100_st_vgg(repeats, output_path="output/"):
    
    args = Namespace(dataset="cifar100",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def curriculum_cifar100_st_vgg(repeats, output_path="output/"):
    args = Namespace(dataset="cifar100",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     transfer_net="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


if __name__ == "__main__":
    
    output_path = "output/"
    num_repeats = 30
    
#    import datasets.cifar100
#    
#    dataset = datasets.cifar100.Cifar100(False)
    
    
    #case 2 & 3
#    vanilla_cifar10_st_vgg(num_repeats, output_path=output_path)
#    curriculum_cifar10_st_vgg(num_repeats, output_path=output_path)
#    vanilla_cifar100_st_vgg(num_repeats, output_path=output_path)
#     curriculum_cifar100_st_vgg(num_repeats, output_path=output_path)
    
    # case 1
    # curriculum_small_mammals(num_repeats, output_path=output_path)
    # vanilla_small_mammals(num_repeats, output_path=output_path)
    # anti_curriculum_small_mammals(num_repeats, output_path=output_path)
    # random_small_mammals(num_repeats, output_path=output_path)


###subset 16###

    ###cld###
    #case_1_cld_curriculum16(num_repeats, output_path=output_path)
    #case_1_cld_anti16(num_repeats, output_path=output_path)
    #case_1_cld_random16(num_repeats, output_path=output_path)

    ###kdn###
    #case_1_kDN_curriculum16(num_repeats, output_path=output_path)
    #case_1_kDN_anti16(num_repeats, output_path=output_path)
    #case_1_kDN_random16(num_repeats, output_path=output_path)

    ###dcp###
    #case_1_DCP_curriculum16(num_repeats, output_path=output_path)
    #case_1_DCP_anti16(num_repeats, output_path=output_path)
    #case_1_DCP_random16(num_repeats, output_path=output_path)

###subset 17###
    ###cld###
    #case_1_cld_curriculum17(num_repeats, output_path=output_path)
    #case_1_cld_anti17(num_repeats, output_path=output_path)
    #case_1_cld_random17(num_repeats, output_path=output_path)

    ###kdn###
    #case_1_kDN_curriculum17(num_repeats, output_path=output_path)
    #case_1_kDN_anti17(num_repeats, output_path=output_path)
    #case_1_kDN_random17(num_repeats, output_path=output_path)

    ###dcp
    #case_1_DCP_curriculum17(num_repeats, output_path=output_path)
    #case_1_DCP_anti17(num_repeats, output_path=output_path)
    #case_1_DCP_random17(num_repeats, output_path=output_path)

###subset 19###
    ###cld###
    #case_1_cld_curriculum19(num_repeats, output_path=output_path)
    #case_1_cld_anti19(num_repeats, output_path=output_path)
    #case_1_cld_random19(num_repeats, output_path=output_path)

    ###kdn###
    #case_1_kDN_curriculum19(num_repeats, output_path=output_path)
    #case_1_kDN_anti19(num_repeats, output_path=output_path)
    #case_1_kDN_random19(num_repeats, output_path=output_path)

    ###dcp###
    #case_1_DCP_curriculum19(num_repeats, output_path=output_path)
    #case_1_DCP_anti19(num_repeats, output_path=output_path)
    #case_1_DCP_random19(num_repeats, output_path=output_path)

###subset 14###
    ###cld###
    #case_1_cld_curriculum14(num_repeats, output_path=output_path)
    #case_1_cld_anti14(num_repeats, output_path=output_path)
    #case_1_cld_random14(num_repeats, output_path=output_path)

    ###kdn###
    #case_1_kDN_curriculum14(num_repeats, output_path=output_path)
    #case_1_kDN_anti14(num_repeats, output_path=output_path)
    #case_1_kDN_random14(num_repeats, output_path=output_path)

    ####dcp###
    #case_1_DCP_curriculum14(num_repeats, output_path=output_path)
    #case_1_DCP_anti14(num_repeats, output_path=output_path)
    #case_1_DCP_random14(num_repeats, output_path=output_path)


    # case_1_MV(num_repeats, output_path=output_path)
    # case_1_CB(num_repeats, output_path=output_path)
    # case_1_N1(num_repeats, output_path=output_path)
    # case_1_N2(num_repeats, output_path=output_path)
    # case_1_F1(num_repeats, output_path=output_path)
    # case_1_F2(num_repeats, output_path=output_path)
    # case_1_F3(num_repeats, output_path=output_path)
    # case_1_F4(num_repeats, output_path=output_path)
    # case_1_LSC(num_repeats, output_path=output_path)
    # case_1_LSR(num_repeats, output_path=output_path)
    # case_1_Harmfulness(num_repeats, output_path=output_path)
    # case_1_Usefulness(num_repeats, output_path=output_path)
