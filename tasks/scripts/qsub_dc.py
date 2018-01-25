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
             hidden_layer_size, min_ngram, max_ngram, min_df, i, dc_type,
             data_name,
             # output_size,
             random_split):
    code_path = "/export/b02/fwang/.work/dc/code/tf_mlp_dc.py"
    base_dir = "/export/b02/fwang/.work/dc/workspace/" + dc_type + "/" + \
               data_name + "/"
    data_dir = "/export/b02/fwang/.work/dc/datasets/" + dc_type + "/" + data_name + "/"
    index_data_name = "data.json"
    mtx_data_name = data_name + "_" + \
                    str(min_ngram) + "_" + str(max_ngram) + "_" + str(
        min_df) + ".mtx"
    report_name = data_name + "_" + str(i) + ".report.md"
    report_dir = "/export/b02/fwang/.work/dc/report/"
    job_name = "job_" + str(i) + "_" + dc_type + "_" + data_name + "_" + str(
        learning_rate) + "_" + str(dropout_rate) + "_" + str(
        lambda_) + "_" + str(batch_size) + "_" + str(
        hidden_layer_size) + "_" + str(min_ngram) + "_" + str(
        max_ngram) + "_" + str(min_df)
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
             "  --model_dir=" + model_dir + \
             "  --data_dir=" + data_dir + \
             "  --index_data_name=" + index_data_name + \
             "  --mtx_data_name=" + mtx_data_name + \
             "  --report_dir=" + report_dir + \
             "  --report_name=" + report_name + \
             "  --min_ngram=" + str(min_ngram) + \
             "  --max_ngram=" + str(max_ngram) + \
             "  --min_df=" + str(min_df) + \
             "  --min_df=" + str(min_df) + \
             "  --random_split=" + random_split + \
             "  --learning_rate=" + str(learning_rate) + \
             "  --dropout_rate=" + str(dropout_rate) + \
             "  --lambda_=" + str(lambda_) + \
             "  --batch_size=" + str(batch_size) + \
             "  --hidden_layer_size=" + str(hidden_layer_size)
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
    random_split = input()

    learning_rates = [0.0001]
    dropout_rates = [0, 0.5]
    lambdas = [0, 0.0001]
    batch_sizes = [32]
    hidden_layer_sizes = [100]
    min_ngrams = [1]
    max_ngrams = [3]
    min_dfs = [50]

    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for lambda_ in lambdas:
                for batch_size in batch_sizes:
                    for hidden_layer_size in hidden_layer_sizes:
                        for min_ngram in min_ngrams:
                            for max_ngram in max_ngrams:
                                for min_df in min_dfs:
                                    if lambda_ > 0 and dropout_rate > 0 \
                                            or lambda_ == 0 and dropout_rate == 0:
                                        pass
                                    else:
                                        make_job(learning_rate,
                                                 dropout_rate,
                                                 lambda_,
                                                 batch_size,
                                                 hidden_layer_size,
                                                 min_ngram, max_ngram, min_df,
                                                 i,
                                                 dc_type,
                                                 data_name,
                                                 # output_size,
                                                 random_split)


if __name__ == "__main__":
    main()
