import csv
import os
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing


class gnb:

    def __init__(self):
        self.label_test = []
        self.pendapatan_numerik = []
        self.nilai_numerik = []
        self.rating_numerik = []
        self.genre_numerik = []
        self.pendapatan_test_numerik = []
        self.nilai_test_numerik = []
        self.rating_test_numerik = []
        self.genre_test_numerik = []
        self.data_test_numerik = []
        self.data_numerik_ar = []
        self.label_numerik = []
        self.data = []
        self.x_values = []
        self.y = []
        self.tampilan_numerik_datapelatihan = []
        self.mean = None
        self.var = None
        self.mean_var = []
        self.mean_var2 = []
        self.test = '.\Data Testing.csv'
        self.datatest = []
        self.tampung = []
        self.probtampung = []
        self.probakhir = []
        self.tampunglagi = []
        self.evidence = []
        self.klasifikasi = []

    def load(self, filename='.\datasetCopy.csv'):
        lines = csv.reader(open(filename))
        dataset = list(lines)
        rating = []
        genre = []
        nilai = []
        pendapatan = []
        label = []
        le = preprocessing.LabelEncoder()
        for i in range(len(dataset)):
            dataset[i] = [str(x) for x in dataset[i]]
        temp = np.array(dataset)
        self.data = temp[:, 1:6]
        for item in range(len(self.data)):
            genre.append(self.data[item, 0])
        for item in range(len(self.data)):
            rating.append(self.data[item, 1])
        for item in range(len(self.data)):
            nilai.append(self.data[item, 2])
        for item in range(len(self.data)):
            pendapatan.append(self.data[item, 3])
        for item in range(len(self.data)):
            label.append(self.data[item, 4])
        self.genre_numerik = list(le.fit_transform(genre))
        self.rating_numerik = list(le.fit_transform(rating))
        self.nilai_numerik = list(le.fit_transform(nilai))
        self.pendapatan_numerik = list(le.fit_transform(pendapatan))
        self.label_numerik = list(le.fit_transform(label))
        return genre

    def transform_data(self):
        pd.set_option('display.max_rows', None)
        self.x_values = pd.DataFrame({'Genre': self.genre_numerik, 'Rating': self.rating_numerik,
                                      'Score': self.nilai_numerik, 'Pendapatan': self.pendapatan_numerik})
        self.y = pd.Series(self.label_numerik)
        self.data_numerik_ar = self.x_values.to_numpy()
        return [self.x_values, self.y]

    def loadtestdata(self):
        lines2 = csv.reader(open(self.test))
        datasettest = list(lines2)
        genre_test = []
        rating_test = []
        nilai_test = []
        pendapatan_test = []
        le = preprocessing.LabelEncoder()
        for i in range(len(datasettest)):
            datasettest[i] = [str(x) for x in datasettest[i]]
        tempo = np.array(datasettest)
        self.datatest = tempo[:, 0:4]
        for item in range(len(self.datatest)):
            genre_test.append(self.datatest[item, 0])
        for item in range(len(self.datatest)):
            rating_test.append(self.datatest[item,1])
        for item in range(len(self.datatest)):
            nilai_test.append(self.datatest[item, 2])
        for item in range(len(self.datatest)):
            pendapatan_test.append(self.datatest[item, 3])
        self.label_test = np.random.randint(0, 9, (len(tempo), 4))
        self.genre_test_numerik = list(le.fit_transform(genre_test))
        self.rating_test_numerik = list(le.fit_transform(rating_test))
        self.nilai_test_numerik = list(le.fit_transform(nilai_test))
        self.pendapatan_test_numerik = list(le.fit_transform(pendapatan_test))
        return self.datatest

    def transform_datatest(self):
        genre_t = []
        rating_t = []
        nilai_t = []
        pendapatan_t = []
        id = []
        for i in range(len(self.datatest)):
            for j in range(len(self.data)):
                for k in range(len(self.data[j, 0:4])):
                    if self.datatest[i][k] == self.data[j][k]:
                        self.label_test[i][k] = self.data_numerik_ar[j][k]
                        id.append([i, j, k])
                    elif int(self.datatest[i][3]) >= 18000000:
                        self.label_test[i][3] = random.randint(80, 90)

        for item in range(len(self.label_test)):
            genre_t.append(self.label_test[item, 0])
        for item in range(len(self.label_test)):
            rating_t.append(self.label_test[item,1])
        for item in range(len(self.label_test)):
            nilai_t.append(self.label_test[item, 2])
        for item in range(len(self.label_test)):
            pendapatan_t.append(self.label_test[item, 3])

        self.tampilan_numerik_datapelatihan = pd.DataFrame({'Genre': genre_t, 'Rating': rating_t,
                                      'Score': nilai_t, 'Pendapatan': pendapatan_t})
        return self.tampilan_numerik_datapelatihan

    def ratavarian(self):
        m = self.x_values.groupby(by=self.y).mean()
        self.mean = np.array(m)
        v = self.x_values.groupby(by=self.y).var()
        self.var = np.array(v)
        return [m, v]

    def meanvar(self):
        m_row = 0
        mean = 0
        v_row = 0
        var = 0
        for i in range(len(self.mean)):
            m_row = self.mean[i]
            v_row = self.var[i]
            for a, b in enumerate(m_row):
                mean = b
                var = v_row[a]
                self.mean_var.append([mean, var])
        self.mean_var2 = np.split(np.array(self.mean_var), 2)
        return self.mean_var

    def probgaussian(self, m, v, value):
        pi = np.pi
        equ_1 = 1 / (np.sqrt(2 * pi * v))
        denom = 2 * v
        numerator = pow((value - m), 2)
        expo = np.exp(-(numerator / denom))
        prob = equ_1 * expo
        return prob

    def hasil(self):
        for i in self.label_test:
            for j in range(len(self.mean_var2)):
                k = self.mean_var2[j]
                for item in range(len(k)):
                    t1 = k[item][0]
                    t2 = k[item][1]
                    t3 = i[item]
                    self.tampung.append([t1, t2, t3])
                    self.probtampung.append([self.probgaussian(t1, t2, t3)])
        return self.probtampung

    def finalprob(self):
        simpan = []
        simpan2 = []
        n_kelas = [np.count_nonzero(np.array(self.label_numerik) == 0),
                   np.count_nonzero(np.array(self.label_numerik) == 1)]
        self.tampunglagi = np.vsplit(np.array(self.probtampung), len(self.label_test))
        for i in self.tampunglagi:
            simpan2 = np.vsplit(i, 2)
            self.evidence.append(np.prod(simpan2[0]) * (n_kelas[0] / sum(n_kelas)) +
                                 np.prod(simpan2[1]) * (n_kelas[1] / sum(n_kelas)))
        for item in range(len(self.tampunglagi)):
            simpan = np.vsplit(self.tampunglagi[item], 2)
            self.probakhir.append([np.prod(simpan[0]) * (n_kelas[0] / sum(n_kelas)) / self.evidence[item],
                                   np.prod(simpan[1]) * (n_kelas[1] / sum(n_kelas)) / self.evidence[item]])

        for prob in self.probakhir:
            if prob[0] > prob[1]:
                self.klasifikasi.append("Gagal")
            else:
                self.klasifikasi.append("Sukses")
        return np.array(self.klasifikasi)


gnb = gnb()
print(gnb.load())
b = gnb.transform_data()
gnb.loadtestdata()
c = gnb.transform_datatest()
a = gnb.ratavarian()
gnb.meanvar()
gnb.hasil()
d = gnb.finalprob()
print("=============DATA PELATIHAN=============")
print(b[0])
print("    Label")
print(b[1])
print("=============Mean(0) dan Varians(1)=============")
print("Data dengan label Gagal")
print(a[0])
print("Data dengan label Sukses")
print(a[1])
print("=============DATA PENGUJIAN=============")
print(c)
print("=============HASIL KLASIFIKASI=============")
print(pd.Series(d))
os.system("pause")
