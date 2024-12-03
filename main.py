import Parser
from trainer import Trainer
from Const import device
from StockData import StockDataGenerator
from Util import make_file
from Normalizer import Normalizer
from DataLoader import data_loader
from Preprocessor import prepare
from Student import study


def main():
    mode = "study"

    print("Device: ", device)
    print("--------------------------- STEP 0 CONSTANT --------------------------")
    Parser.print_params()

    print("--------------------------- STEP 1 DATA GENERATOR --------------------")
    generator = StockDataGenerator()
    data_set = generator.allGenerateData()  # ndarray
    # data_set = generator.dummy()

    print("--------------------------- STEP 2 NORMALIZE -------------------------")
    # if need_normalize:
    #     data, target, scaler = normalize(data, target)
    normalizer = Normalizer()

    print("--------------------------- STEP 3 PREPARE DATA ----------------------")
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare(data_set, need_encode=False)

    print("--------------------------- STEP 4 MAKE MODEL ------------------------")
    # feature_size = train_x.shape[2]

    print("--------------------------- STEP 5 MAKE DATALOADER -------------------")
    train_loader, valid_loader, test_loader = data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y)

    trainer = Trainer(train_loader, valid_loader, test_loader)
    if mode == "study":
        print("--------------------------- STEP 6 STUDYING ---------------------------")
        study(trainer)
    elif mode == "train":
        print("--------------------------- STEP 6 TRAINING ---------------------------")

    # ---------------------------------------------------------------------------
    # ------------- STEP 7: SHOW RESULT --------------------------------------------
    # ---------------------------------------------------------------------------
    # print("STEP 7 SHOW RESULT")
    # pred = trainer.eval(lstm_model)
    # pred = pred[:, :, 0]
    #
    #
    # real = data_set.test_target
    # output = []
    # for index, _ in enumerate(real):
    #     diff_index = index - input_window
    #     if len(pred) <= diff_index:
    #         break
    #     if diff_index < 0:
    #         output.append(0)
    #         continue
    #     else:
    #         data = real[index - 1] * (1 + (pred[diff_index]/100))
    #         output.append(data[0])
    # output = np.array(output)
    # print("Output shape: ", output.shape)
    # showTemp(real, np.array(output))
    # printt(real, np.array(output), test_y, pred)


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    make_file()
    main()