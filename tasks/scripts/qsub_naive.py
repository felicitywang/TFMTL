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


def make_job(learning_rate, dropout_rate, lambda_, batch_size,
             hidden_layer_size, i, dc_type, data_name,
             output_size, model_path):
    code_path = "/export/b02/fwang/.work/dc/code/naive.py"
    base_dir = "/export/b02/fwang/.work/dc/workspace/" + dc_type + \
               "/SSTb_IMDB/naive/" + \
               data_name + "/"
    data_dir = "/export/b02/fwang/.work/dc/datasets/" + dc_type + \
               "/SSTb_IMDB/" + data_name + "/"
    report_name = data_name + "_" + str(i) + ".transfer.report.md"
    report_dir = "/export/b02/fwang/.work/dc/report/SSTb_IMDB/"
    job_name = "job_" + str(i) + "_" + dc_type + "_" + data_name + "_" + str(
        learning_rate) + "_" + str(dropout_rate) + "_" + str(
        lambda_) + "_" + str(batch_size) + "_" + str(
        hidden_layer_size)
    model_dir = base_dir + "models/" + job_name + "/"
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
        os.stat(report_dir)
    except:
        os.mkdir(report_dir)
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
#$ -l mem_free=20G,ram_free=20G'\n\
#$ -pe smp 2\n\
#$ -V\n\
\n\
\n\
source /home/fwang/.bashrc \n\
mkdir " + model_dir + " \n\
\n\
/export/b02/fwang/.work/dc/venv/bin/python3.4 " + \
             code_path + \
             "  --model_dir=" + model_dir + \
             "  --data_dir=" + data_dir + \
             "  --report_dir=" + report_dir + \
             "  --report_name=" + report_name + \
             "  --learning_rate=" + str(learning_rate) + \
             "  --dropout_rate=" + str(dropout_rate) + \
             "  --lambda_=" + str(lambda_) + \
             "  --batch_size=" + str(batch_size) + \
             "  --hidden_layer_size=" + str(hidden_layer_size) + \
             "  --output_size=" + str(output_size) + \
             "  --model_path=" + str(model_path)

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
    output_size = int(input())
    # whether to split
    # random_split = input()
    model_path = input()

    learning_rates = [0.0001]
    dropout_rates = [0.5]
    lambdas = [0]
    batch_sizes = [32]
    hidden_layer_sizes = [100]

    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for lambda_ in lambdas:
                for batch_size in batch_sizes:
                    for hidden_layer_size in hidden_layer_sizes:
                        if lambda_ > 0 and dropout_rate > 0 \
                                or lambda_ == 0 and dropout_rate == 0:
                            pass
                        else:
                            make_job(learning_rate,
                                     dropout_rate,
                                     lambda_,
                                     batch_size,
                                     hidden_layer_size,
                                     i,
                                     dc_type,
                                     data_name,
                                     output_size,
                                     model_path)


if __name__ == "__main__":
    main()
