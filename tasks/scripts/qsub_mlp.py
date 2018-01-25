# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import subprocess
import sys


def make_job(dc_type, data_name, i):
    code_path = "/export/b02/fwang/mlvae/tasks/code/test_mlp.py"
    base_dir = "/export/b02/fwang/mlvae/tasks/workspace/" + dc_type + "/" + \
               data_name + "/"
    data_dir = "/export/b02/fwang/mlvae/tasks/datasets/" + \
               dc_type + "/" + data_name + "/"
    job_name = "job_" + str(i) + "_" + dc_type + "_" + data_name
    model_dir = base_dir + "models/" + job_name + "/"
    best_model_dir = model_dir + "best_model/"
    model_path = best_model_dir + "model.ckpt"
    try:
        os.stat(base_dir)
    except:
        os.mkdir(base_dir)
    try:
        os.stat(base_dir + "models/")
    except:
        os.mkdir(base_dir + "models/")
    try:
        os.stat(model_dir)
    except:
        os.mkdir(model_dir)
    try:
        os.stat(best_model_dir)
    except:
        os.mkdir(best_model_dir)
    try:
        os.stat("jobs/")
    except:
        os.mkdir("jobs/")
    output = "#!/bin/sh \n\
#$ -cwd \n\
#$ -o " + model_dir + "log \n\
#$ -e " + model_dir + "err \n\
#$ -m eas\n\
#$ -M cnfxwang@gmail.com\n\
#$ -l gpu=1,mem_free=80G,ram_free=80G,hostname='b*|c*'\n\
#$ -pe smp 2\n\
#$ -V\n\
#$ -q g.q\n\
\n\
\n\
source /home/fwang/.bashrc \n\
mkdir " + model_dir + " \n\
\n\
CUDA_VISIBLE_DEVICES=`free-gpu` python3 " + code_path + \
             "  --data_dir=" + data_dir + \
             "  --model_path=" + model_path
    #  "  --learning_rate=" + str(learning_rate) + \
    #  "  --dropout_rate=" + str(dropout_rate) + \
    #  "  --lambda_=" + str(lambda_) + \
    #  "  --batch_size=" + str(batch_size) + \
    #  "  --hidden_layer_size=" + str(hidden_layer_size)
    # "  --output_size=" + str(output_size)

    print(output)

    f_out = open("jobs/" + job_name + ".sh", "w")
    f_out.write(output)
    f_out.close()

    subprocess.call(["qsub", "jobs/" + job_name + ".sh"])


def main():
    # integer in the front of job name to differentiate jobs
    i = sys.argv[1]
    # document classification type
    # sentiment, emotion, or dialogue
    dc_type = input()
    # data name e.g. IMDB, SSTb, etc.
    data_name = input()
    # output size (# labels)
    # output_size = int(input())
    # whether to split

    make_job(dc_type, data_name, i)


if __name__ == "__main__":
    main()
