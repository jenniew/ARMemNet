
export OMP_NUM_THREADS=10

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master yarn \
 --deploy-mode client \
 --driver-memory 40g \
 --executor-memory 40g \
 --executor-cores 32 \
 --conf spark.driver.host=i01 \
 --num-executors 2 \
 --files /home/nvkvs/ARMemNet-ding/data_utils.py \
 train_mem_model_zoo.py hdfs://192.168.111.24:9000/tmp/data 4096 10 /home/nvkvs/ARMemNet-ding/model 32
