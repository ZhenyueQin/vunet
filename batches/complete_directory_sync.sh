#!/usr/bin/env bash

#rsync -a --ignore-existing ../. liu162@bracewell.hpc.csiro.au:/home/liu162/Zhenyue/Project-Wukong-VAE-Study/Kristiadi-Generative-Models

rsync -av --exclude=restorer_log --exclude=outs --exclude=log --update . liu162@bracewell.hpc.csiro.au:/flush2/liu162/Zhenyue-Qin/Project-Wukong-V-U-Net --exclude='*.png'