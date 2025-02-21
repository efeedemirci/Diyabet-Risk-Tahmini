from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np
import sys

# Veri seti dosyaları
DATASETS = {
    "Orijinal": "diabetes.csv",
    "Normalize Edilmiş": "normalized.csv",
    "Dengesiz": "imbalanced.csv",
    "Gürültülü": "noised.csv"
}

class MLApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MLApp, self).__init__()
        uic.loadUi('ml_interface.ui', self)  # Qt Designer'da oluşturulan arayüz dosyası

        # UI öğelerini bağlama
        self.datasetComboBox = self.findChild(QtWidgets.QComboBox, 'datasetComboBox')
        self.modelComboBox = self.findChild(QtWidgets.QComboBox, 'modelComboBox')
        self.kfoldCheckBox = self.findChild(QtWidgets.QCheckBox, 'kfoldCheckBox')
        self.runButton = self.findChild(QtWidgets.QPushButton, 'runButton')
        self.predictButton = self.findChild(QtWidgets.QPushButton, 'predictButton')
        self.inputLayout = self.findChild(QtWidgets.QVBoxLayout, 'inputLayout')
        self.confusionMatrixWidget = self.findChild(QtWidgets.QWidget, 'confusionMatrixWidget')
        self.dataTableWidget = self.findChild(QTableWidget, 'dataTableWidget')

        self.runButton.clicked.connect(self.run_model)
        self.predictButton.clicked.connect(self.predict_with_user_input)

        # Veri seti ve model seçeneklerini doldurma
        self.datasetComboBox.addItems(DATASETS.keys())
        self.modelComboBox.addItems(["Lojistik Regresyon", "KNN", "Random Forest", "Karar Ağacı"])

        self.input_entries = {}
        self.init_input_fields()

    def init_input_fields(self):
        # Nitelik giriş alanlarını oluşturma
        try:
            data_sample = pd.read_csv("diabetes.csv").drop(columns=["Outcome"]).iloc[0]
            for feature in data_sample.index:
                label = QtWidgets.QLabel(feature)
                entry = QtWidgets.QLineEdit()
                self.inputLayout.addWidget(label)
                self.inputLayout.addWidget(entry)
                self.input_entries[feature] = entry
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Nitelik giriş alanları oluşturulamadı: {e}")

    def load_and_process_data(self, dataset_name):
        try:
            original_data = pd.read_csv(DATASETS["Orijinal"])
            X = original_data.drop(columns=["Outcome"])
            y = original_data["Outcome"]

            if dataset_name == "Normalize Edilmiş":
                scaler = MinMaxScaler()
                X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                normalized_data = pd.concat([X_normalized, y], axis=1)
                normalized_data.to_csv("normalized.csv",index=False)
                self.show_data_in_table(normalized_data)
                return normalized_data, None

            elif dataset_name == "Dengesiz":
                class_0 = original_data[original_data["Outcome"] == 0]
                class_1 = original_data[original_data["Outcome"] == 1]
                class_1_downsampled = resample(class_1, replace=False, n_samples=len(class_0) // 2, random_state=42)
                imbalanced_data = pd.concat([class_0, class_1_downsampled])
                self.show_data_in_table(imbalanced_data)
                imbalanced_data.to_csv("imbalanced.csv",index=False)

                smote = SMOTE(random_state=42)
                X_smote, y_smote = smote.fit_resample(imbalanced_data.drop(columns=["Outcome"]),imbalanced_data["Outcome"])
                balanced_data = pd.concat([pd.DataFrame(X_smote, columns=X.columns), pd.Series(y_smote, name="Outcome")], axis=1)
                balanced_data.to_csv("balanced.csv",index=False)

                self.show_data_in_table(imbalanced_data)
                return imbalanced_data, balanced_data
                

            elif dataset_name == "Gürültülü":
                # Gürültü eklenecek örnek sayısını hesapla (toplam örnek sayısının %5'i)
                noise_count = int(len(X) * 0.05)
                # Rastgele örnekler seç
                random_indices = np.random.choice(X.index, size=noise_count, replace=False)
                # Gürültü eklenecek örnekleri seç ve bu örneklere gürültü ekle
                X_noised = X.copy()
                for index in random_indices:
                    noise = np.random.normal(0, 0.1, X.loc[index].shape)  # Her bir örnek için gürültü
                    X_noised.loc[index] += noise  # Gürültüyü ekle

                # Gürültü eklenmiş veri ve etiketleri birleştir
                noised_data = pd.concat([X_noised, y], axis=1)
                self.show_data_in_table(noised_data)
                noised_data.to_csv("noised.csv",index=False)

                # Gürültü eklenmiş olan verileri denoised_data'ya al
                X_denoised = X_noised[(np.abs(X_noised - X) < 0.1).all(axis=1)]
                y_denoised = y[X_denoised.index]
                denoised_data = pd.concat([X_denoised, y_denoised], axis=1)
                denoised_data.to_csv("denoised.csv",index=False)

                # Veriyi tablo halinde göster
                self.show_data_in_table(denoised_data)
                return noised_data, denoised_data

            else:
                self.show_data_in_table(original_data)
                return original_data, None

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri işlenirken bir hata oluştu: {e}")
            return None

    def show_data_in_table(self, data):
        self.dataTableWidget.clear()  # Mevcut tabloyu temizle
        self.dataTableWidget.setRowCount(data.shape[0])
        self.dataTableWidget.setColumnCount(data.shape[1])
        self.dataTableWidget.setHorizontalHeaderLabels(data.columns)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.dataTableWidget.setItem(i, j, QTableWidgetItem(str(data.iat[i, j])))

    def run_model(self):
        dataset_name = self.datasetComboBox.currentText()
        use_kfold = self.kfoldCheckBox.isChecked()
        model_name = self.modelComboBox.currentText()

        if not dataset_name or not model_name:
            QMessageBox.critical(self, "Hata", "Lütfen bir veri seti ve model seçin!")
            return

        data, additional_data = self.load_and_process_data(dataset_name)
        if data is None:
            return

        def evaluate_model(data, title):
            try:
                X = data.drop(columns=["Outcome"])
                y = data["Outcome"]
                labels = sorted(y.unique())

                if model_name == "Lojistik Regresyon":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100)
                elif model_name == "Karar Ağacı":
                    model = DecisionTreeClassifier(max_depth=5)


                if use_kfold:
                    kf = StratifiedKFold(n_splits=5)
                    accuracies = []
                    confusion_matrices = []
                    kfold_results = ""

                    for i, (train_index, test_index) in enumerate(kf.split(X, y),start=1):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracies.append(accuracy_score(y_test, y_pred))
                        conf = confusion_matrix(y_test, y_pred)
                        confusion_matrices.append(conf)

                        # Her iterasyon için karışıklık matrisini ve sınıflandırma raporunu kfoldTextEdit alanına ekliyoruz
                        classification = classification_report(y_test, y_pred, labels=labels)
                        kfold_results += f"K-Fold İterasyon {i}\nKarışıklık Matrisi:\n{conf}\nSınıflandırma Raporu:\n{classification}\n\n"

                    # Sonuçları kfoldTextEdit alanına yazdırıyoruz
                    self.kfoldTextEdit.setPlainText(kfold_results)  # K-Fold sonuçlarını yazdırıyoruz
                    self.show_data_in_table(data)

                    cm = np.mean(confusion_matrices, axis=0)
                    accuracy = np.mean(accuracies)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred, labels=labels)
                    classification = classification_report(y_test, y_pred, labels=labels)
                    self.show_data_in_table(data)

                QMessageBox.information(
                    self,
                    f"Sonuçlar: {title}",
                    f"Model: {model_name}\nVeri Seti: {dataset_name}\nK-Fold: {'Evet' if use_kfold else 'Hayır'}\nKarışıklık Matrisi:\n {cm}\nAccuracy: {accuracy:.2f}\n\n{classification}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Model çalıştırılırken bir hata oluştu: {e}")

        evaluate_model(data, "İlk Veri")
        if additional_data is not None:
            evaluate_model(additional_data, "İşlem Sonrası Veri")

    def predict_with_user_input(self):
        dataset_name = self.datasetComboBox.currentText()
        model_name = self.modelComboBox.currentText()

        if not dataset_name or not model_name:
            QMessageBox.critical(self, "Hata", "Lütfen bir veri seti ve model seçin!")
            return

        try:
            if dataset_name == "Gürültülü":
                data = pd.read_csv("denoised.csv")
            elif dataset_name == "Dengesiz":
                data = pd.read_csv("balanced.csv")
            else:
                data = pd.read_csv(DATASETS[dataset_name])
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri seti yüklenemedi: {e}")
            return

        try:
            X = data.drop(columns=["Outcome"])
        except KeyError:
            QMessageBox.critical(self, "Hata", "Veri seti formatı hatalı! 'Outcome' sütunu bulunamadı.")
            return

        user_input = []
        for feature, entry in self.input_entries.items():
            try:
                user_input.append(float(entry.text()))
            except ValueError:
                QMessageBox.critical(self, "Hata", f"{feature} için geçerli bir değer girin!")
                return

        if model_name == "Lojistik Regresyon":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == "Karar Ağacı":
            model = DecisionTreeClassifier(max_depth=5)


        X_train, _, y_train, _ = train_test_split(X, data["Outcome"], test_size=0.2, random_state=None)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba([user_input])[0]

        # Sonuçları belirle
        result_diyabet_var = probabilities[1]  # Diyabet Var için olasılık
        result_diyabet_yok = probabilities[0]  # Diyabet Yok için olasılık

        # Kullanıcıya sonucu göster
        result_message = (f"Diyabet Yok olasılığı: {result_diyabet_yok:.2f}\n"
                          f"Diyabet Var olasılığı: {result_diyabet_var:.2f}")

        QMessageBox.information(self, "Tahmin Sonucu", result_message)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec_())
