from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
from data_utils import *
from AR_mem.config import Config
from AR_mem.model import Model
from time import time
import tensorflow as tf
from zoo.common import set_core_number


# get dataset from given preprocessed_dir parallelly by spark
def get_datasets_from_dir_spark(sc, preprocessed_dir, batch_size, train_cells=1.0, valid_cells=0, test_cells=0):
    # logger
    logger = logging.getLogger()

    # load preprocessed files from dir & get total rows
    preprocessed_files = os.listdir(preprocessed_dir)
    n_preprocessed_files = len(preprocessed_files)

    # split train / valid / test set
    if train_cells <= 1.0:
        n_train_set = round(n_preprocessed_files * train_cells)
    else:
        n_train_set = int(train_cells)

    if valid_cells <= 1.0:
        n_valid_set = round(n_preprocessed_files * valid_cells)
    else:
        n_valid_set = int(valid_cells)

    if test_cells <= 1.0:
        n_test_set = round(n_preprocessed_files * test_cells)
    else:
        n_test_set = int(test_cells)

    # split by index
    idx_cells = np.random.permutation(n_preprocessed_files)
    idx_train = idx_cells[:n_train_set]
    idx_valid = idx_cells[n_train_set:n_train_set + n_valid_set]
    idx_test = idx_cells[n_train_set + n_valid_set:n_train_set + n_valid_set + n_test_set]

    train_files = [preprocessed_files[j] for j in idx_train]
    valid_files = [preprocessed_files[j] for j in idx_valid]
    test_files = [preprocessed_files[j] for j in idx_test]

    assert n_train_set + n_valid_set + n_test_set <= n_preprocessed_files

    # define train_set properties
    n_rows_per_file = np.load(os.path.join(preprocessed_dir, train_files[0]))['X'].shape[0]
    n_total_rows = n_train_set * n_rows_per_file

    # log dataset info
    logger.info('')
    logger.info('Dataset Summary')
    logger.info(' - Used {:6d} cells of {:6d} total cells ({:2.2f}%)'.format(n_train_set + n_valid_set + n_test_set,
                                                                             n_preprocessed_files, (
                                                                                         n_train_set + n_valid_set + n_test_set) / n_preprocessed_files * 100))
    logger.info(' - Train Dataset: {:6d} cells ({:02.2f}% of used cells)'.format(n_train_set, n_train_set / (
                n_train_set + n_valid_set + n_test_set) * 100))
    logger.info(' - Valid Dataset: {:6d} cells ({:02.2f}% of used cells)'.format(n_valid_set, n_valid_set / (
                n_train_set + n_valid_set + n_test_set) * 100))
    logger.info(' - Test Dataset : {:6d} cells ({:02.2f}% of used cells)'.format(n_test_set, n_test_set / (
                n_train_set + n_valid_set + n_test_set) * 100))
    logger.info('')
    logger.info('Trainset Summary')
    logger.info(' - Row / Cell: {:9d} rows / cell'.format(n_rows_per_file))
    logger.info(' - Train Cell: {:9d} cells'.format(n_train_set))
    logger.info(' - Total Rows: {:9d} rows'.format(n_total_rows))
    logger.info(' - Batch Size: {:9d} rows / batch'.format(batch_size))
    logger.info(' - Batch Step: {:9d} batches / epoch'.format(math.ceil(n_total_rows / batch_size)))
    logger.info('')

    train_data = sc.parallelize(train_files).\
        map(lambda file: read_npz_file(preprocessed_dir, file)).\
        flatMap(lambda data_seq: get_feature_label_list(data_seq))
    val_data = sc.parallelize(valid_files). \
        map(lambda file: read_npz_file(preprocessed_dir, file)).\
        flatMap(lambda data_seq: get_feature_label_list(data_seq))
    test_data = sc.parallelize(test_files). \
        map(lambda file: read_npz_file(preprocessed_dir, file)).\
        flatMap(lambda data_seq: get_feature_label_list(data_seq))

    return train_data, val_data, test_data


if __name__ == "__main__":

    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    model_dir = sys.argv[4]

    # For tuning
    if len(sys.argv) > 5:
        core_num = int(sys.argv[5])
    else:
        core_num = 4
    if len(sys.argv) > 6:
        thread_num = int(sys.argv[6])
    else:
        thread_num = 10

    config = Config()
    config.data_path = data_path
    config.latest_model=False
    config.batch_size = batch_size

    # init or get SparkContext
    sc = init_nncontext()
    
    # tuning
    set_core_number(core_num)

    # create train data
    # train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = \
    #     load_agg_selected_data_mem_train(data_path=config.data_path,
    #                                x_len=config.x_len,
    #                                y_len=config.y_len,
    #                                foresight=config.foresight,
    #                                cell_ids=config.train_cell_ids,
    #                                dev_ratio=config.dev_ratio,
    #                                test_len=config.test_len,
    #                                seed=config.seed)

    # config.batch_size is useless as we force get_datasets_from_dir return the entire data
    # train_X, train_Y, train_M, valid_X, valid_Y, valid_M, _, _, _ =\
    #     get_datasets_from_dir(sc, config.data_path, config.batch_size,
    #                       train_cells=config.num_cells_train,
    #                       valid_cells=config.num_cells_valid,
    #                       test_cells=config.num_cells_test)[0]
    #
    # dataset = TFDataset.from_ndarrays([train_X, train_M, train_Y], batch_size=batch_size,
    #                                   val_tensors=[valid_X, valid_M, valid_Y],)
    #
    # model = Model(config, dataset.tensors[0], dataset.tensors[1], dataset.tensors[2])

    train_rdd, val_rdd, test_rdd = \
        get_datasets_from_dir_spark(sc, config.data_path, config.batch_size,
                                    train_cells=config.num_cells_train,
                                    valid_cells=config.num_cells_valid,
                                    test_cells=config.num_cells_test)

    dataset = TFDataset.from_rdd(train_rdd,
                                 features=[(tf.float32, [10, 8]), (tf.float32, [77, 8])],
                                 labels=(tf.float32, [8]),
                                 batch_size=config.batch_size,
                                 val_rdd=val_rdd)

    model = Model(config, dataset.tensors[0][0], dataset.tensors[0][1], dataset.tensors[1])

    optimizer = TFOptimizer.from_loss(model.loss, Adam(config.lr),
                                      metrics={"rse": model.rse, "smape": model.smape, "mae": model.mae},
                                      model_dir=model_dir,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                                                    intra_op_parallelism_threads=thread_num)
                                      )

    start_time = time()
    optimizer.optimize(end_trigger=MaxEpoch(num_epochs))
    end_time = time()

    print("Elapsed training time {} secs".format(end_time - start_time))


