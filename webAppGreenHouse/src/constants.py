from header_names import *
DIR_MODELLI = 'modelli'
DIR_ESITI = 'esiti'
DIR_DATI = 'dati'
CSV_BASE_NAME='Dataset_sens_'
data_header_index = 'timestamp_normalizzato'
data_headers = {'base': data_header_base,
               'esteso': data_header_esteso,
               'meno_tempo':data_header_con_meno_tempo,
               'senza_distanza':data_header_senza_distanza,
               'senza_evja':data_header_senza_evja,
                'data_header_solo_altri_dati':data_header_solo_altri_dati,
                'soloTempo':data_header_soloTempo}
data_header = data_header_base
###########################################################################
DROPOUT_LABEL ='dropout'
NUNITS_LABEL = 'nunits'
NUMFEATURES_LABEL = 'numfeatures'
EPOCHS_LABEL = 'epochs'
SENSOR_LABEL = 'sensor'
NUMLAYERS_LABEL = 'num_layers'
XTRAIN_LABEL = 'x_train'
YTRAIN_LABEL = 'y_train'
XTEST_LABEL = 'x_test'
YTEST_LABEL = 'y_test'
DROPOUTS_VALUES_LABEL = 'dropout_values'
NUNITS_VALUES_LABEL = 'nunits_values'
EPOCHS_VALUES_LABEL = 'epochs_values'
MAXNUMLAYERS_LABEL = 'max_num_layers'
BATCH_SIZE_LABEL = 'batch_size'
VALIDATION_SPLIT_LABEL = 'validation_split'
MAX_TIME_LAG_LABEL = 'max_time_lag'
TIME_LAG_LABEL = 'time_lag'
PERFORMANCES_LABEL = 'performances'
THIS_MODEL_LABEL = 'thismodel'
BEST_ONE_LABEL = 'bestOne'
EVALUATION_METRIC_LABEL = 'evalMetric'
LSTM_MODEL_LABEL = 'LSTM_model'
DATADESCR_LABEL = 'dataset'
DATASET_SPLIT_RANDOM_LABEL = 'shuffle'
UNDERSCORE ='_'
TWOUNDERSCORE ='__'
CSV_EXTENSION = '.csv'
SUFFISSO_MODELLO_KERAS='.keras'

#
CHOSEN_LOSS = 'mean_absolute_error'
MAPE_LABEL = 'mape'
MAE_LABEL = 'mae'
MSE_LABEL = 'mse'
R2_LABEL = 'r2_score'
#
LOG_SHAPE_OF_MSG = 'Shape of '
LOG_DATA_WORD_MSG = 'data '
LOG_TRAIN_WORD_MSG = 'train '
LOG_TEST_WORD_MSG = 'test '
LOG_RESULT_WORD_MSG = 'result '
LOG_VALUES_WORD_MSG = 'values '
LOG_BEST_ONE_MSG = 'il migliore ora risulta quello con: '
