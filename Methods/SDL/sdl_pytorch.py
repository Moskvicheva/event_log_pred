import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import os

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = y  # y is already one-hot encoded

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleModel(nn.Module):
    def __init__(self, input_sizes, num_activities):
        super(SimpleModel, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(input_size, input_size, padding_idx=0) for input_size in input_sizes])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(sum(input_sizes), num_activities)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embeds = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embeds, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def transform_data(log, columns):
    num_activities = len(log.values[log.activity]) + 1  # unique values
    col_num_vals = {}
    for col in columns:
        if col == log.activity:
            col_num_vals[col] = num_activities
        else:
            col_num_vals[col] = log.contextdata[col].max() + 2

    inputs = []
    for _ in range(len(columns) * log.k - len(log.ignoreHistoryAttributes) * log.k):
        inputs.append([])
    outputs = []
    for row in log.contextdata.iterrows():
        row = row[1]
        i = 0
        for attr in columns:
            if attr not in log.ignoreHistoryAttributes:
                for k in range(log.k):
                    inputs[i].append(row[attr + "_Prev%i" % k])
                    i += 1
        outputs.append(row[log.activity])

    # Convert outputs to categorical using PyTorch
    outputs = torch.tensor(outputs, dtype=torch.long)
    outputs = F.one_hot(outputs, num_classes=num_activities).float()

    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])

    return inputs, outputs, col_num_vals


# Пример использования обновленной функции transform_data в контексте learn_model

def learn_model(log, attributes, epochs, early_stop):
    num_activities = len(log.values[log.activity]) + 1
    
    input_sizes = []
    for attr in attributes:
        if attr not in log.ignoreHistoryAttributes and attr != log.time and attr != log.trace:
            for k in range(log.k):
                input_sizes.append(len(log.values[attr]) + 1)

    model = SimpleModel(input_sizes, num_activities)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.004)
    optimizer = optim.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.004)
    
    x, y, col_num_vals = transform_data(log, [a for a in attributes if a != log.time and a != log.trace])
    
    if len(y) < 10:
        validation_split = 0
    else:
        validation_split = 0.2

    x = np.stack(x, axis=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, random_state=42)
    
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
            torch.save(model.state_dict(), f'tmp/model_{epoch:03d}-{val_loss:.2f}.pt')
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model



def train(log, epochs, early_stop):
    # train(data_object) - type Data
    # return learn_model(log, log.attributes(), epochs, early_stop)
  return learn_model(log.logfile, log.logfile.attributes(), epochs, early_stop)


def update(model, log):
    inputs, expected, _ = transform_data(log, [a for a in log.logfile.attributes() if a != log.time and a != log.trace])
    model.fit(inputs, y=expected,
              validation_split=0,
              verbose=0,
              batch_size=32,
              epochs=10)
    return model



def test(model, log):
    model.eval()  # Устанавливаем модель в режим оценки
    inputs, expected, _ = transform_data(log.logfile, [a for a in log.logfile.attributes() if a != log.logfile.time and a != log.logfile.trace])
    
    inputs = np.stack(inputs, axis=1)
    expected = torch.tensor(expected, dtype=torch.float)

    test_dataset = CustomDataset(inputs, expected)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predict_vals = []
    predict_probs = []
    expected_vals = []
    expected_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            predicted_probs = torch.softmax(outputs, dim=1).gather(1, predicted.unsqueeze(1)).squeeze()

            actual = torch.argmax(batch_y, 1)
            actual_probs = torch.softmax(outputs, dim=1).gather(1, actual.unsqueeze(1)).squeeze()

            predict_vals.extend(predicted.cpu().numpy())
            predict_probs.extend(predicted_probs.cpu().numpy())
            expected_vals.extend(actual.cpu().numpy())
            expected_probs.extend(actual_probs.cpu().numpy())

    result = zip(expected_vals, predict_vals, predict_probs, expected_probs)
    return list(result)



def test_and_update(logs, model):
    results = []
    i = 0
    for t in logs:
        print(i, "/", len(logs))
        i += 1
        log = logs[t]["data"]
        results.extend(test(log, model))

        inputs, expected, _ = transform_data(log.logfile, [a for a in log.logfile.attributes() if a != log.logfile.time and a != log.logfile.trace])
        model.fit(inputs, y=expected,
                  validation_split=0,
                  verbose=0,
                  batch_size=32,
                  epochs=10)
    return results


def test_and_update_retain(test_logs, model, train_log):
    train_x, train_y, _ = transform_data(train_log.logfile, [a for a in train_log.attributes() if a != train_log.time and a != train_log.trace])

    results = []
    i = 0
    for t in test_logs:
        print(i, "/", len(test_logs))
        i += 1
        test_log = test_logs[t]["data"]
        results.extend(test(test_log, model))
        test_x, test_y, _ = transform_data(test_log.logfile, [a for a in test_log.attributes() if a != test_log.time and a != test_log.trace])
        for j in range(len(train_x)):
            train_x[j] = np.hstack([train_x[j], test_x[j]])
        train_y = np.concatenate((train_y, test_y))
        model.fit(train_x, y=train_y,
                  validation_split=0.2,
                  verbose=0,
                  batch_size=32,
                  epochs=1)
    return results


def test_and_update_full(test_log, model, train_logs):
    results = test(test_log, model)

    train = train_logs[0]
    train_x, train_y, _ = transform_data(train, [a for a in train.attributes() if a != train.time and a != train.trace])
    for t_idx in range(1, len(train_logs)):
        t = train_logs[t_idx]
        test_x, test_y, _ = transform_data(t, [a for a in t.attributes() if a != t.time and a != t.trace])
        for j in range(len(train_x)):
            train_x[j] = np.hstack([train_x[j], test_x[j]])
        train_y = np.concatenate((train_y, test_y))
    model.fit(train_x, y=train_y,
              validation_split=0.2,
              verbose=1,
              batch_size=32,
              epochs=1)
    return results


    # results = []
    # update_range = 1000
    # for i in range(0, len(test_x[0]), update_range):
    #     single_input = [el[i:i+update_range] for el in test_x]
    #     predictions = model.predict(single_input)
    #     predict_vals = np.argmax(predictions, axis=1)
    #     predict_probs = predictions[np.arange(predictions.shape[0]), predict_vals]
    #     if update_range == 1:
    #         expected_vals = np.argmax(test_y[i])
    #         results.append((expected_vals, predict_vals[0], predict_probs[0]))
    #     else:
    #         expected_vals = np.argmax(test_y[i:i+update_range], axis=1)
    #         results.extend(zip(expected_vals, predict_vals, predict_probs))
    #

    #
    #     model.fit(x=train_x, y=train_y,
    #               validation_split=0.2,
    #               verbose=0,
    #               batch_size=32,
    #               epochs=10)
    # return results


