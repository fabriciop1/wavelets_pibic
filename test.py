# -*- coding: utf8 -*-

import cv2
import numpy as np
import pywt
import os.path
import csv
import random
import mlpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from Util import Util
import timeit
 
def write_csv(f, folder):  
    c = csv.writer(open(f, "wb"))
    
    SEPARATOR = ";"
    BASE_PATH = folder

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                c.writerow(["%s%s%d" % (abs_path, SEPARATOR, label)])
            label = label + 1

def wavelets_lda(f, holdouts, n_classes, training_sample):
    
    nn = mlpy.KNN(k=1)                                                                      # Classificador 1NN
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto', n_components=60)   # Método de extração de características - LDA
    pca_fast=  mlpy.PCAFast(k=168)                                                          # Método de extração de características - PCA (implementação de Sharma e Paliwal usando algoritmo de ponto fixo)
    wavelet = pywt.Wavelet('haar')                                                          # Wavelet de Haar
    gnb = GaussianNB()                                                                      # Classificador Gaussian Naive Bayes
    svm = mlpy.LibLinear(solver_type='l2r_l2loss_svc_dual', C=1)                            # Classificador Support Vector Machines

    files = []
    
    with open(f, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter= ';')
        for line in reader:
            files.append([line[0], line[1]])
        
    files = np.array(files)
    taxas_acerto_svm = []
    taxas_acerto_knn = []
    taxas_acerto_gnb = []
    
    for i in range(holdouts):
        training = []
        test = []
        training_imgs = []
        test_imgs = []
    
        for i in range(0, n_classes):
            temp_arr_rows = np.where(files[:, 1].astype(int) == i)
            treino_arr = random.sample(files[temp_arr_rows], training_sample)

            for i in treino_arr:
                training.append(i)
        
            test_arr = [j for j in files[temp_arr_rows].tolist() if j not in np.asarray(treino_arr).tolist()]
        
            for i in test_arr:
                test.append(i)
        
        training = np.array(training)
        test = np.array(test)
        
        for row in training:
            img = cv2.imread(row[0], 0)
            coeffs = pywt.wavedec2(img, wavelet, level=3)
            training_imgs.append(coeffs[0].flatten())

        for row in test:
            img = cv2.imread(row[0], 0)
            coeffs = pywt.wavedec2(img, wavelet, level=3)
            test_imgs.append(coeffs[0].flatten())
        
        training_imgs = np.array(training_imgs)
        test_imgs = np.array(test_imgs)
        
        #pca_fast.learn(training_imgs)
        #y_fast = pca_fast.transform(training_imgs)
        #test_fast = pca_fast.transform(test_imgs)
        
        lda.fit(training_imgs, training[0:training.shape[0], len(training[0])-1:len(training[0])].flatten())
        ldaT = lda.transform(training_imgs)
        z = lda.transform(test_imgs)
        
        svm.learn(ldaT, training[0:training.shape[0], len(training[0])-1:len(training[0])].flatten())                                     
        preds = svm.pred(z)

        taxas_acerto_svm.append(util.getAccuracy(test, preds.astype(str)))
        
        gnb.fit(ldaT, training[0:training.shape[0], len(training[0])-1:len(training[0])].flatten())
        preds = gnb.predict(z)

        taxas_acerto_gnb.append(util.getAccuracy(test, preds.astype(str)))
        
        nn.learn(ldaT, training[0:training.shape[0], len(training[0])-1:len(training[0])].flatten())
        preds = nn.pred(z)

        taxas_acerto_knn.append(util.getAccuracy(test, preds.astype(str)))

    return taxas_acerto_svm, taxas_acerto_gnb, taxas_acerto_knn

if __name__ == "__main__":
    
    start_time = timeit.default_timer()
    util = Util()
    f = "faces95.csv"
    folder = "faces95/"
    write_csv(f, folder)
    
    taxas_acerto_svm, taxas_acerto_gnb, taxas_acerto_knn = wavelets_lda(f, 10, 72, 10)

    print "Wavelet Transform Level 3 + LDA"
    print "\nSupport Vector Machines: ", np.array(taxas_acerto_svm), "\nMedia: ", np.mean(np.array(taxas_acerto_svm)), "Desvio Padrao: ", np.std(np.array(taxas_acerto_svm))
    print "\nGaussian Naive Bayes: ", np.array(taxas_acerto_gnb), "\nMedia: ", np.mean(np.array(taxas_acerto_gnb)), "Desvio Padrao: ", np.std(np.array(taxas_acerto_gnb))
    print "\n1NN: ", np.array(taxas_acerto_knn), "\nMedia: ", np.mean(np.array(taxas_acerto_knn)), "Desvio Padrao: ", np.std(np.array(taxas_acerto_knn))
    print "\nTime: ", (timeit.default_timer() - start_time) / 60.0, " mins"
