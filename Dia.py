import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sqlite3
import json

class DiabeticPatientMonitor:
    def __init__(self, patient_id, patient_name, diabetes_type, config_file="config.json"):
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.diabetes_type = diabetes_type  # 1, 2 ou autre
        self.config = self.load_config(config_file)
        self.data = {
            'timestamp': [],
            'blood_glucose': [],
            'glucose_trend': [],
            'insulin_dose': [],
            'carbs_intake': [],
            'physical_activity': [],
            'heart_rate': [],
            'blood_pressure_systolic': [],
            'blood_pressure_diastolic': [],
            'weight': [],
            'ketones': [],
            'hypo_symptoms': [],
            'hyper_symptoms': []
        }
        self.meal_log = []
        self.insulin_log = []
        self.setup_database()
        
    def load_config(self, config_file):
        """Charge la configuration à partir d'un fichier JSON"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Configuration par défaut pour le diabète
            return {
                "alert_thresholds": {
                    "hypoglycemia": 70,       # mg/dL
                    "hyperglycemia": 180,     # mg/dL
                    "severe_hyperglycemia": 250,  # mg/dL
                    "ketoacidosis_risk": 1.5,     # mmol/L (cétones)
                    "hypo_symptoms_severity": 7   # sur une échelle de 1-10
                },
                "target_ranges": {
                    "fasting_glucose": "80-130 mg/dL",
                    "postprandial_glucose": "80-180 mg/dL",
                    "hba1c": "<7%",
                    "blood_pressure": "<140/90 mmHg"
                },
                "email_alerts": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": "",
                    "recipient_emails": []
                },
                "prediction_settings": {
                    "window_size": 10,
                    "forecast_hours": 4
                }
            }
    
    def setup_database(self):
        """Configure la base de données SQLite"""
        self.conn = sqlite3.connect(f'diabetic_patient_{self.patient_id}.db')
        self.cursor = self.conn.cursor()
        
        # Création des tables spécifiques au diabète
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS glucose_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                blood_glucose REAL,
                measurement_context TEXT,  -- à jeun, post-prandial, etc.
                glucose_trend TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS insulin_doses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                insulin_type TEXT,
                units REAL,
                injection_site TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                carbs_grams REAL,
                food_description TEXT,
                estimated_glucose_impact REAL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                alert_type TEXT,
                message TEXT,
                severity TEXT
            )
        ''')
        
        self.conn.commit()
    
    def simulate_glucose_measurement(self, context="random"):
        """Simule une mesure de glycémie basée sur le contexte"""
        # Valeurs basées sur le contexte et le type de diabète
        if context == "fasting":
            base_value = 100 if self.diabetes_type == 2 else 130
            variability = 20 if self.diabetes_type == 2 else 40
        elif context == "postprandial":
            base_value = 140 if self.diabetes_type == 2 else 180
            variability = 40 if self.diabetes_type == 2 else 60
        else:  # random
            base_value = 120 if self.diabetes_type == 2 else 160
            variability = 50 if self.diabetes_type == 2 else 80
        
        # Ajouter de la variabilité
        glucose_value = max(50, min(400, random.normalvariate(base_value, variability)))
        
        # Déterminer la tendance
        if len(self.data['blood_glucose']) > 0:
            last_reading = self.data['blood_glucose'][-1]
            if glucose_value > last_reading + 15:
                trend = "↑↑"  # Forte augmentation
            elif glucose_value > last_reading + 5:
                trend = "↑"   # Légère augmentation
            elif glucose_value < last_reading - 15:
                trend = "↓↓"  # Forte diminution
            elif glucose_value < last_reading - 5:
                trend = "↓"   # Légère diminution
            else:
                trend = "→"   # Stable
        else:
            trend = "→"
        
        return round(glucose_value, 1), trend
    
    def log_glucose_measurement(self, context="random"):
        """Enregistre une mesure de glycémie"""
        glucose_value, trend = self.simulate_glucose_measurement(context)
        current_time = datetime.now()
        
        # Ajouter aux données en mémoire
        self.data['timestamp'].append(current_time)
        self.data['blood_glucose'].append(glucose_value)
        self.data['glucose_trend'].append(trend)
        
        # Sauvegarder dans la base de données
        self.cursor.execute('''
            INSERT INTO glucose_measurements (timestamp, blood_glucose, measurement_context, glucose_trend)
            VALUES (?, ?, ?, ?)
        ''', (current_time, glucose_value, context, trend))
        
        self.conn.commit()
        
        # Vérifier les alertes
        self.check_glucose_alerts(glucose_value)
        
        return glucose_value, trend
    
    def log_insulin_dose(self, insulin_type, units, injection_site="abdomen"):
        """Enregistre une dose d'insuline"""
        current_time = datetime.now()
        
        # Sauvegarder dans la base de données
        self.cursor.execute('''
            INSERT INTO insulin_doses (timestamp, insulin_type, units, injection_site)
            VALUES (?, ?, ?, ?)
        ''', (current_time, insulin_type, units, injection_site))
        
        self.conn.commit()
        
        # Ajouter aux données en mémoire
        self.data['insulin_dose'].append(units)
        
        return True
    
    def log_meal(self, carbs_grams, food_description=""):
        """Enregistre un repas"""
        current_time = datetime.now()
        
        # Estimer l'impact sur la glycémie (simplifié)
        estimated_impact = carbs_grams / 10  # Impact approximatif en mg/dL
        
        # Sauvegarder dans la base de données
        self.cursor.execute('''
            INSERT INTO meals (timestamp, carbs_grams, food_description, estimated_glucose_impact)
            VALUES (?, ?, ?, ?)
        ''', (current_time, carbs_grams, food_description, estimated_impact))
        
        self.conn.commit()
        
        # Ajouter aux données en mémoire
        self.data['carbs_intake'].append(carbs_grams)
        
        return True
    
    def log_symptoms(self, symptom_type, severity, notes=""):
        """Enregistre des symptômes d'hypo ou d'hyperglycémie"""
        current_time = datetime.now()
        
        if symptom_type == "hypo":
            self.data['hypo_symptoms'].append(severity)
            if severity >= self.config['alert_thresholds']['hypo_symptoms_severity']:
                self.send_alert("hypo_symptoms", f"Symptômes d'hypoglycémie sévères (niveau {severity}/10)")
        else:
            self.data['hyper_symptoms'].append(severity)
            if severity >= self.config['alert_thresholds']['hypo_symptoms_severity']:
                self.send_alert("hyper_symptoms", f"Symptômes d'hyperglycémie sévères (niveau {severity}/10)")
        
        return True
    
    def check_glucose_alerts(self, glucose_value):
        """Vérifie les alertes basées sur la glycémie"""
        thresholds = self.config['alert_thresholds']
        
        if glucose_value < thresholds['hypoglycemia']:
            self.send_alert("hypoglycemia", f"Hypoglycémie détectée: {glucose_value} mg/dL")
        
        elif glucose_value > thresholds['severe_hyperglycemia']:
            self.send_alert("severe_hyperglycemia", f"Hyperglycémie sévère: {glucose_value} mg/dL")
        
        elif glucose_value > thresholds['hyperglycemia']:
            self.send_alert("hyperglycemia", f"Hyperglycémie: {glucose_value} mg/dL")
        
        return True
    
    def send_alert(self, alert_type, message):
        """Envoie une alerte via multiple canaux"""
        # Déterminer la sévérité
        if alert_type == "severe_hyperglycemia" or "hypo" in alert_type:
            severity = "high"
        else:
            severity = "medium"
        
        # Enregistrer l'alerte dans la base de données
        current_time = datetime.now()
        self.cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (current_time, alert_type, message, severity))
        
        self.conn.commit()
        
        # Afficher dans la console
        print(f"🔴 ALERTE: {message} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Envoyer par email si configuré
        if self.config['email_alerts']['enabled']:
            self.send_email_alert(message, severity)
    
    def send_email_alert(self, message, severity):
        """Envoie une alerte par email"""
        try:
            sender_email = self.config['email_alerts']['sender_email']
            sender_password = self.config['email_alerts']['sender_password']
            recipient_emails = self.config['email_alerts']['recipient_emails']
            
            if not sender_email or not recipient_emails:
                return
            
            # Création du message
            email_message = MIMEMultipart()
            email_message['From'] = sender_email
            email_message['To'] = ", ".join(recipient_emails)
            email_message['Subject'] = f"Alerte Diabète - {self.patient_name} - {severity.upper()}"
            
            body = f"""
            Alerte pour le patient: {self.patient_name}
            Type de diabète: {self.diabetes_type}
            Niveau de sévérité: {severity.upper()}
            Message: {message}
            Horodatage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Veuillez prendre les mesures appropriées.
            """
            email_message.attach(MIMEText(body, 'plain'))
            
            # Connexion au serveur SMTP et envoi
            with smtplib.SMTP(self.config['email_alerts']['smtp_server'], 
                             self.config['email_alerts']['smtp_port']) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_emails, email_message.as_string())
                
            print("📧 Email d'alerte envoyé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi de l'email: {str(e)}")
    
    def predict_glucose_trend(self, hours=4):
        """Prédit l'évolution de la glycémie"""
        if len(self.data['blood_glucose']) < self.config['prediction_settings']['window_size']:
            return "Données insuffisantes pour la prédiction"
        
        # Préparer les données pour la prédiction
        window_size = self.config['prediction_settings']['window_size']
        glucose_data = self.data['blood_glucose'][-window_size:]
        
        # Créer un modèle de régression linéaire simple
        X = np.arange(len(glucose_data)).reshape(-1, 1)
        y = np.array(glucose_data)
        
        # Entraîner le modèle
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédire les prochaines heures
        future_X = np.arange(len(glucose_data), len(glucose_data) + hours).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Vérifier les alertes potentielles
        alerts = []
        for i, pred in enumerate(predictions):
            if pred < self.config['alert_thresholds']['hypoglycemia']:
                alerts.append(f"Hypoglycémie prévue dans {i+1} heures ({pred:.1f} mg/dL)")
            elif pred > self.config['alert_thresholds']['severe_hyperglycemia']:
                alerts.append(f"Hyperglycémie sévère prévue dans {i+1} heures ({pred:.1f} mg/dL)")
        
        if alerts:
            return "\n".join(alerts)
        
        return "Aucune anomalie glycémique prévue dans les prochaines heures"
    
    def generate_diabetes_report(self, hours=24):
        """Génère un rapport complet pour le diabète"""
        # Récupérer les données de la base de données
        query = f"""
        SELECT timestamp, blood_glucose, measurement_context, glucose_trend 
        FROM glucose_measurements 
        WHERE timestamp >= datetime('now', '-{hours} hours')
        ORDER BY timestamp
        """
        df_glucose = pd.read_sql_query(query, self.conn)
        
        if df_glucose.empty:
            return "Aucune donnée de glycémie disponible pour cette période"
        
        # Calculer les statistiques
        mean_glucose = df_glucose['blood_glucose'].mean()
        min_glucose = df_glucose['blood_glucose'].min()
        max_glucose = df_glucose['blood_glucose'].max()
        
        # Compter les épisodes d'hypo et d'hyperglycémie
        hypo_episodes = len(df_glucose[df_glucose['blood_glucose'] < self.config['alert_thresholds']['hypoglycemia']])
        hyper_episodes = len(df_glucose[df_glucose['blood_glucose'] > self.config['alert_thresholds']['hyperglycemia']])
        severe_hyper_episodes = len(df_glucose[df_glucose['blood_glucose'] > self.config['alert_thresholds']['severe_hyperglycemia']])
        
        # Générer le rapport
        report = f"""
📊 RAPPORT DIABÈTE - {self.patient_name}
Type de diabète: {self.diabetes_type}
Période: {hours} heures
----------------------------------------
Valeurs glycémiques:
- Moyenne: {mean_glucose:.1f} mg/dL
- Minimum: {min_glucose:.1f} mg/dL
- Maximum: {max_glucose:.1f} mg/dL

Épisodes:
- Hypoglycémie (<{self.config['alert_thresholds']['hypoglycemia']} mg/dL): {hypo_episodes}
- Hyperglycémie (>{self.config['alert_thresholds']['hyperglycemia']} mg/dL): {hyper_episodes}
- Hyperglycémie sévère (>{self.config['alert_thresholds']['severe_hyperglycemia']} mg/dL): {severe_hyper_episodes}

Prédiction:
{self.predict_glucose_trend()}

Objectifs thérapeutiques:
- Glycémie à jeun: {self.config['target_ranges']['fasting_glucose']}
- Glycémie post-prandiale: {self.config['target_ranges']['postprandial_glucose']}
- HbA1c: {self.config['target_ranges']['hba1c']}
- Pression artérielle: {self.config['target_ranges']['blood_pressure']}
        """
        
        return report
    
    def plot_glucose_data(self, hours=24):
        """Crée un graphique des données de glycémie"""
        # Récupérer les données
        query = f"""
        SELECT timestamp, blood_glucose 
        FROM glucose_measurements 
        WHERE timestamp >= datetime('now', '-{hours} hours')
        ORDER BY timestamp
        """
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            print("Aucune donnée à afficher pour cette période.")
            return
        
        # Convertir le timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['blood_glucose'], 'b-', label='Glycémie (mg/dL)')
        
        # Ajouter les lignes de référence
        plt.axhline(y=self.config['alert_thresholds']['hypoglycemia'], color='r', linestyle='--', label='Seuil hypoglycémie')
        plt.axhline(y=self.config['alert_thresholds']['hyperglycemia'], color='orange', linestyle='--', label='Seuil hyperglycémie')
        plt.axhline(y=self.config['alert_thresholds']['severe_hyperglycemia'], color='purple', linestyle='--', label='Seuil hyperglycémie sévère')
        
        plt.ylabel('Glycémie (mg/dL)')
        plt.title(f'Surveillance Glycémique - {self.patient_name}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarder et afficher le graphique
        plt.savefig(f'glucose_monitoring_{self.patient_id}_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        plt.show()

class DiabetesMonitoringGUI:
    def __init__(self, monitor):
        self.monitor = monitor
        self.root = tk.Tk()
        self.root.title(f"Surveillance Diabète - {monitor.patient_name}")
        self.setup_gui()
        
    def setup_gui(self):
        """Configure l'interface graphique"""
        # Cadre principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Informations patient
        ttk.Label(main_frame, text=f"Patient: {self.monitor.patient_name} (Type {self.monitor.diabetes_type})", 
                 font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Affichage des valeurs en temps réel
        self.glucose_var = tk.StringVar(value="Glycémie: -- mg/dL")
        self.trend_var = tk.StringVar(value="Tendance: --")
        self.status_var = tk.StringVar(value="Statut: --")
        
        ttk.Label(main_frame, textvariable=self.glucose_var, font=('Arial', 16, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.trend_var, font=('Arial', 12)).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 12)).grid(row=3, column=0, sticky=tk.W, pady=5)
        
        # Boutons pour différentes actions
        ttk.Button(main_frame, text="Mesure Glycémie", command=self.measure_glucose).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="Enregistrer Repas", command=self.log_meal).grid(row=2, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="Enregistrer Insuline", command=self.log_insulin).grid(row=3, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="Rapport Complet", command=self.show_report).grid(row=4, column=0, pady=10)
        ttk.Button(main_frame, text="Graphique Glycémie", command=self.show_glucose_plot).grid(row=4, column=1, pady=10)
        
        # Zone de texte pour les alertes
        self.alert_text = tk.Text(main_frame, height=10, width=50)
        self.alert_text.grid(row=5, column=0, columnspan=2, pady=10)
        self.alert_text.insert(tk.END, "Aucune alerte pour le moment\n")
        self.alert_text.config(state=tk.DISABLED)
        
        # Configuration de la mise à jour automatique
        self.update_interval = 30000  # 30 secondes
        self.update_display()
        
    def measure_glucose(self):
        """Prend une mesure de glycémie et met à jour l'affichage"""
        glucose_value, trend = self.monitor.log_glucose_measurement()
        self.update_display_values(glucose_value, trend)
    
    def log_meal(self):
        """Ouvre une fenêtre pour enregistrer un repas"""
        meal_window = tk.Toplevel(self.root)
        meal_window.title("Enregistrer un repas")
        
        ttk.Label(meal_window, text="Glucides (g):").grid(row=0, column=0, padx=5, pady=5)
        carbs_entry = ttk.Entry(meal_window)
        carbs_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(meal_window, text="Description:").grid(row=1, column=0, padx=5, pady=5)
        desc_entry = ttk.Entry(meal_window)
        desc_entry.grid(row=1, column=1, padx=5, pady=5)
        
        def submit_meal():
            try:
                carbs = float(carbs_entry.get())
                desc = desc_entry.get()
                self.monitor.log_meal(carbs, desc)
                messagebox.showinfo("Succès", f"Repas enregistré: {carbs}g de glucides")
                meal_window.destroy()
            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre valide pour les glucides")
        
        ttk.Button(meal_window, text="Enregistrer", command=submit_meal).grid(row=2, column=0, columnspan=2, pady=10)
    
    def log_insulin(self):
        """Ouvre une fenêtre pour enregistrer une dose d'insuline"""
        insulin_window = tk.Toplevel(self.root)
        insulin_window.title("Enregistrer une dose d'insuline")
        
        ttk.Label(insulin_window, text="Type d'insuline:").grid(row=0, column=0, padx=5, pady=5)
        insulin_type = ttk.Combobox(insulin_window, values=["Rapide", "Lente", "Mixte", "Basale", "Bolus"])
        insulin_type.grid(row=0, column=1, padx=5, pady=5)
        insulin_type.set("Rapide")
        
        ttk.Label(insulin_window, text="Unités:").grid(row=1, column=0, padx=5, pady=5)
        units_entry = ttk.Entry(insulin_window)
        units_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(insulin_window, text="Site d'injection:").grid(row=2, column=0, padx=5, pady=5)
        site_entry = ttk.Combobox(insulin_window, values=["Abdomen", "Bras", "Cuisse", "Fesse"])
        site_entry.grid(row=2, column=1, padx=5, pady=5)
        site_entry.set("Abdomen")
        
        def submit_insulin():
            try:
                units = float(units_entry.get())
                ins_type = insulin_type.get()
                site = site_entry.get()
                self.monitor.log_insulin_dose(ins_type, units, site)
                messagebox.showinfo("Succès", f"Dose d'insuline enregistrée: {units} unités ({ins_type})")
                insulin_window.destroy()
            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre valide pour les unités")
        
        ttk.Button(insulin_window, text="Enregistrer", command=submit_insulin).grid(row=3, column=0, columnspan=2, pady=10)
    
    def update_display_values(self, glucose_value, trend):
        """Met à jour l'affichage avec les nouvelles valeurs"""
        self.glucose_var.set(f"Glycémie: {glucose_value} mg/dL")
        self.trend_var.set(f"Tendance: {trend}")
        
        # Déterminer le statut
        if glucose_value < self.monitor.config['alert_thresholds']['hypoglycemia']:
            status = "HYPOGLYCÉMIE - Prendre du sucre"
            color = "red"
        elif glucose_value > self.monitor.config['alert_thresholds']['severe_hyperglycemia']:
            status = "HYPERGLYCÉMIE SÉVÈRE - Contacter médecin"
            color = "purple"
        elif glucose_value > self.monitor.config['alert_thresholds']['hyperglycemia']:
            status = "Hyperglycémie - Surveiller"
            color = "orange"
        else:
            status = "Dans la cible"
            color = "green"
        
        self.status_var.set(f"Statut: {status}")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, foreground=color)
        self.status_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        
        self.update_alert_display()
    
    def update_alert_display(self):
        """Met à jour l'affichage des alertes"""
        # Récupérer les alertes de la base de données
        query = "SELECT timestamp, message FROM alerts ORDER BY timestamp DESC LIMIT 5"
        df_alerts = pd.read_sql_query(query, self.monitor.conn)
        
        self.alert_text.config(state=tk.NORMAL)
        self.alert_text.delete(1.0, tk.END)
        
        if df_alerts.empty:
            self.alert_text.insert(tk.END, "Aucune alerte pour le moment\n")
        else:
            for _, row in df_alerts.iterrows():
                self.alert_text.insert(tk.END, f"{row['timestamp']} - {row['message']}\n")
        
        self.alert_text.config(state=tk.DISABLED)
    
    def show_report(self):
        """Affiche un rapport complet"""
        report = self.monitor.generate_diabetes_report(24)
        report_window = tk.Toplevel(self.root)
        report_window.title("Rapport Diabète")
        
        text_area = tk.Text(report_window, width=80, height=20)
        text_area.pack(padx=10, pady=10)
        text_area.insert(tk.END, report)
        text_area.config(state=tk.DISABLED)
    
    def show_glucose_plot(self):
        """Affiche le graphique de glycémie"""
        self.monitor.plot_glucose_data(24)
    
    def update_display(self):
        """Met à jour périodiquement l'affichage"""
        # Pour la démonstration, on prend une mesure aléatoire périodiquement
        if random.random() < 0.3:  # 30% de chance à chaque mise à jour
            self.measure_glucose()
        
        self.root.after(self.update_interval, self.update_display)
    
    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()

# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un moniteur pour un patient diabétique
    diabetic_monitor = DiabeticPatientMonitor("DIA456", "Marie Dupont", diabetes_type=1)
    
    # Lancement de l'interface graphique
    app = DiabetesMonitoringGUI(diabetic_monitor)
    app.run()