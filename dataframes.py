import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


class DataStore:

    def __init__(self, file, names=[], load=True):
        self.filename = file
        self.load = load
        if load:
            self._load()
        else:
            self.names = names
            self._create()

    def _load(self):
        store = pd.HDFStore(self.filename)
        self.names = store.keys()
        self.dataframes = {store[name] for name in self.names}
        store.close()

    def _create(self):
        self.dataframes = {name: pd.DataFrame({}) for name in self.names}
        self._save()

    def _save(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        store = pd.HDFStore(self.filename)
        for name, data in self.dataframes.items():
            store[name] = data
        store.close()

    def add_rows(self, name, data):
        data = pd.DataFrame(data)
        self.dataframes[name] = pd.concat([self.dataframes[name], data])

    def add_col(self, name, data, heading):
        self.dataframes[name][heading] = data

    def get_col(self, name, heading):
        return self.dataframes[name][heading].values

    def get_rows(self, name, col_crit, crit, cols, crit_type='=='):
        if crit_type == '==':
            data = self.dataframes[name].loc[
                self.dataframes[name][col_crit] == crit,
                cols].values
        elif crit_type == '<':
            data = self.dataframes[name].loc[
                self.dataframes[name][col_crit] < crit,
                cols].values
        elif crit_type == '>':
            data = self.dataframes[name].loc[
                self.dataframes[name][col_crit] > crit,
                cols].values
        return data





if __name__ == "__main__":
    DS = DataStore("testing.hdf5", ['a', 'b', 'c'], load=False)
    DS.add_data('a', {'n':np.arange(0, 10), 'nb': np.arange(0, 10)*2})
    print(DS.dataframes['a'].head())

