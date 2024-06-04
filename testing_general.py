import matplotlib.pyplot as plt
import Data
import Methods
import Methods.Camargo
import Methods.Camargo.adapter
import Methods.DiMauro
import Methods.Tax
# import Methods.Tax.adapter
import Methods.Tax.adapter_pytorch
import Predictions.setting
import Predictions.metric
from Predictions.setting.setting import Setting

def plot_statistics(k_array, accuracy, dataset):
    plt.plot(k_array, accuracy)
    plt.title(f'Validation accuracy, {dataset}')
    plt.show()

    
def test_method(method, dataset):
    accuracy_on_k = {}
    data_object = Data.get_data(dataset)
    for k in [1, 5, 15, 20, 30]:
        settings = Setting(k, "train-test", True, False, 70, 10)
        data_object.prepare(settings)
        m = Methods.get_prediction_method(method)
        print('training begin!')
        model = m.train(data_object)
        print('testing begin!')
        results = m.test(model, data_object)
        print(results)
        accuracy = Predictions.metric.ACCURACY.calculate(results)
        print('k = ', k, 'accuracy: ', accuracy)
        accuracy_on_k[k] = accuracy

    print(accuracy_on_k)
    plot_statistics(accuracy_on_k.keys(), accuracy_on_k.values(), dataset)        



