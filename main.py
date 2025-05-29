import os
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure


class NoiseClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes = ["Clean", "Gaussian Noise", "Salt & Pepper Noise", "Speckle Noise"]

    def extract_features(self, image):
        # Convert image to grayscale if it's a color image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Basic statistical features
        mean = np.mean(gray)
        std = np.std(gray)
        var = np.var(gray)

        # Shannon Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        # Edge detection features
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mean = np.mean(np.abs(sobelx) + np.abs(sobely))
        sobel_std = np.std(np.abs(sobelx) + np.abs(sobely))

        # Frequency domain features
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-7)
        freq_mean = np.mean(magnitude_spectrum)
        freq_std = np.std(magnitude_spectrum)

        # GLCM texture features
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()

        # Combine all features
        features = [mean, std, var, entropy, sobel_mean, sobel_std, 
                   freq_mean, freq_std, contrast, dissimilarity, 
                   homogeneity, energy, correlation]

        return np.array(features).reshape(1, -1)

    def train(self, X, y):
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Determine which classes are present in the data
        unique_classes = np.unique(y)
        class_names = [self.classes[i] for i in unique_classes]

        # Calculate additional metrics
        conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_classes)

        # Calculate F1, precision, and recall scores for each class
        f1 = f1_score(y_test, y_pred, labels=unique_classes, average=None)
        precision = precision_score(y_test, y_pred, labels=unique_classes, average=None)
        recall = recall_score(y_test, y_pred, labels=unique_classes, average=None)

        # Calculate average scores
        f1_avg = f1_score(y_test, y_pred, labels=unique_classes, average='weighted')
        precision_avg = precision_score(y_test, y_pred, labels=unique_classes, average='weighted')
        recall_avg = recall_score(y_test, y_pred, labels=unique_classes, average='weighted')

        # Use labels parameter to explicitly specify which labels to include in the report
        report = classification_report(y_test, y_pred, target_names=class_names, labels=unique_classes)

        self.is_trained = True
        return accuracy, report, conf_matrix, class_names, f1, precision, recall, f1_avg, precision_avg, recall_avg

    def predict(self, image):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        # Extract features
        features = self.extract_features(image)

        # Normalize features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        return prediction, probabilities

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.is_trained), f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model, self.scaler, self.is_trained = pickle.load(f)


class ImageProcessor:
    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        if len(image.shape) == 3:
            row, col, ch = image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)
        else:
            row, col = image.shape
            gauss = np.random.normal(mean, sigma, (row, col))
            noisy = image + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
        noisy = np.copy(image)

        # Salt noise
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        if len(image.shape) == 3:
            noisy[salt_mask, :] = 255
        else:
            noisy[salt_mask] = 255

        # Pepper noise
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        if len(image.shape) == 3:
            noisy[pepper_mask, :] = 0
        else:
            noisy[pepper_mask] = 0

        return noisy

    @staticmethod
    def add_speckle_noise(image, intensity=0.1):
        if len(image.shape) == 3:
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            noisy = image + image * gauss * intensity
            return np.clip(noisy, 0, 255).astype(np.uint8)
        else:
            row, col = image.shape
            gauss = np.random.randn(row, col)
            noisy = image + image * gauss * intensity
            return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def generate_training_data(input_dir, output_dir, num_variations=5):
        # Create output directories if they don't exist
        os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "gaussian"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "salt_pepper"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "speckle"), exist_ok=True)

        # Read images from input directory
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for i, img_file in enumerate(image_files):
            img_path = os.path.join(input_dir, img_file)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Save original image to "clean" directory
            clean_path = os.path.join(output_dir, "clean", f"{i}_clean.png")
            cv2.imwrite(clean_path, image)

            # Create Gaussian noise variations
            for j in range(num_variations):
                sigma = np.random.uniform(10, 50)
                noisy = ImageProcessor.add_gaussian_noise(image, sigma=sigma)
                gaussian_path = os.path.join(output_dir, "gaussian", f"{i}_gaussian_{j}.png")
                cv2.imwrite(gaussian_path, noisy)

            # Create Salt & Pepper noise variations
            for j in range(num_variations):
                salt_prob = np.random.uniform(0.01, 0.05)
                pepper_prob = np.random.uniform(0.01, 0.05)
                noisy = ImageProcessor.add_salt_pepper_noise(image, salt_prob, pepper_prob)
                sp_path = os.path.join(output_dir, "salt_pepper", f"{i}_sp_{j}.png")
                cv2.imwrite(sp_path, noisy)

            # Create Speckle noise variations
            for j in range(num_variations):
                intensity = np.random.uniform(0.05, 0.2)
                noisy = ImageProcessor.add_speckle_noise(image, intensity)
                speckle_path = os.path.join(output_dir, "speckle", f"{i}_speckle_{j}.png")
                cv2.imwrite(speckle_path, noisy)

    @staticmethod
    def prepare_dataset(data_dir):
        features = []
        labels = []

        # Read clean images
        clean_dir = os.path.join(data_dir, "clean")
        for img_file in os.listdir(clean_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(clean_dir, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    feature_vector = NoiseClassifier().extract_features(image)
                    features.append(feature_vector.flatten())
                    labels.append(0)  # 0 for clean

        # Read Gaussian noise images
        gaussian_dir = os.path.join(data_dir, "gaussian")
        for img_file in os.listdir(gaussian_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(gaussian_dir, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    feature_vector = NoiseClassifier().extract_features(image)
                    features.append(feature_vector.flatten())
                    labels.append(1)  # 1 for Gaussian

        # Read Salt & Pepper noise images
        sp_dir = os.path.join(data_dir, "salt_pepper")
        for img_file in os.listdir(sp_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(sp_dir, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    feature_vector = NoiseClassifier().extract_features(image)
                    features.append(feature_vector.flatten())
                    labels.append(2)  # 2 for Salt & Pepper

        # Read Speckle noise images
        speckle_dir = os.path.join(data_dir, "speckle")
        for img_file in os.listdir(speckle_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(speckle_dir, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    feature_vector = NoiseClassifier().extract_features(image)
                    features.append(feature_vector.flatten())
                    labels.append(3)  # 3 for Speckle

        return np.array(features), np.array(labels)


class NoiseClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Noise Classifier")
        self.root.geometry("1200x800")

        self.classifier = NoiseClassifier()
        self.image_processor = ImageProcessor()

        self.current_image = None
        self.original_image = None
        self.tk_image = None

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for image display
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image display frame
        self.image_frame = ttk.LabelFrame(left_frame, text="Image", padding=5)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Image control frame
        image_control_frame = ttk.Frame(left_frame)
        image_control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(image_control_frame, text="Import Image", command=self.import_image).pack(side=tk.LEFT, padx=5)

        # Add noise frame
        noise_frame = ttk.LabelFrame(image_control_frame, text="Add Noise")
        noise_frame.pack(side=tk.LEFT, padx=5)

        self.noise_type = tk.StringVar(value="gaussian")
        ttk.Radiobutton(noise_frame, text="Gaussian", variable=self.noise_type, value="gaussian").pack(side=tk.LEFT)
        ttk.Radiobutton(noise_frame, text="Salt & Pepper", variable=self.noise_type, value="salt_pepper").pack(side=tk.LEFT)
        ttk.Radiobutton(noise_frame, text="Speckle", variable=self.noise_type, value="speckle").pack(side=tk.LEFT)

        ttk.Button(noise_frame, text="Add Noise", command=self.add_noise).pack(side=tk.LEFT, padx=5)

        # Right frame for results and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Classification results frame
        result_frame = ttk.LabelFrame(right_frame, text="Classification Results", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_label = ttk.Label(result_frame, text="Not classified yet", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

        # Probability chart frame
        self.plot_frame = ttk.Frame(result_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create chart
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Classification control frame
        classify_frame = ttk.Frame(right_frame)
        classify_frame.pack(fill=tk.X, pady=5)

        ttk.Button(classify_frame, text="Classify Image", command=self.classify_image).pack(side=tk.LEFT, padx=5)

        # Model control frame
        model_frame = ttk.LabelFrame(right_frame, text="Model Controls", padding=5)
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Button(model_frame, text="Generate Training Data", 
                  command=self.generate_training_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Train Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Save Model", 
                  command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Load Model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def import_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )

        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                self.current_image = self.original_image.copy()
                self.display_image(self.current_image)
                self.status_var.set(f"Image imported: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image: {str(e)}")

    def display_image(self, image):
        if image is None:
            return

        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Adjust size to fit display frame
        h, w = image_rgb.shape[:2]
        max_height = 500
        max_width = 600

        # Calculate scale to maintain aspect ratio
        scale = min(max_width / w, max_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)

        # Resize image
        resized = cv2.resize(image_rgb, (new_width, new_height))

        # Convert to Tkinter format
        pil_image = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)

        # Display image
        self.image_label.config(image=self.tk_image)

    def add_noise(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please import an image first")
            return

        noise_type = self.noise_type.get()

        try:
            if noise_type == "gaussian":
                self.current_image = self.image_processor.add_gaussian_noise(self.original_image.copy())
                self.status_var.set("Gaussian noise added")
            elif noise_type == "salt_pepper":
                self.current_image = self.image_processor.add_salt_pepper_noise(self.original_image.copy())
                self.status_var.set("Salt & Pepper noise added")
            elif noise_type == "speckle":
                self.current_image = self.image_processor.add_speckle_noise(self.original_image.copy())
                self.status_var.set("Speckle noise added")

            self.display_image(self.current_image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not add noise: {str(e)}")

    def classify_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please import an image first")
            return

        if not self.classifier.is_trained:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        self.status_var.set("Classifying...")

        # Use thread to avoid freezing the UI
        threading.Thread(target=self._classify_thread).start()

    def _classify_thread(self):
        try:
            prediction, probabilities = self.classifier.predict(self.current_image)

            # Update UI in the main thread
            self.root.after(0, lambda: self._update_results(prediction, probabilities))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification error: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Classification error"))

    def _update_results(self, prediction, probabilities):
        # Update result label
        result_text = f"Result: {self.classifier.classes[prediction]}"
        self.result_label.config(text=result_text)

        # Update probability chart
        self.ax.clear()
        bars = self.ax.bar(self.classifier.classes, probabilities * 100)

        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', rotation=0)

        self.ax.set_ylabel('Probability (%)')
        self.ax.set_title('Classification Probability')
        self.ax.set_ylim(0, 105)  # Set y limit to make room for labels
        plt.xticks(rotation=15)
        plt.tight_layout()

        self.canvas.draw()
        self.status_var.set(f"Classified as: {self.classifier.classes[prediction]}")

    def generate_training_data(self):
        input_dir = filedialog.askdirectory(title="Select directory with clean images")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="Select output directory for training data")
        if not output_dir:
            return

        self.status_var.set("Generating training data...")

        # Use thread to avoid freezing the UI
        threading.Thread(target=self._generate_data_thread, 
                        args=(input_dir, output_dir)).start()

    def _generate_data_thread(self, input_dir, output_dir):
        try:
            self.image_processor.generate_training_data(input_dir, output_dir)
            self.root.after(0, lambda: self.status_var.set("Training data generation completed"))
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                                                         "Training data generation completed"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                          f"Error generating data: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error generating training data"))

    def train_model(self):
        data_dir = filedialog.askdirectory(title="Select directory with training data")
        if not data_dir:
            return

        self.status_var.set("Training model...")

        # Use thread to avoid freezing the UI
        threading.Thread(target=self._train_model_thread, args=(data_dir,)).start()

    def _train_model_thread(self, data_dir):
        try:
            # Prepare data
            X, y = self.image_processor.prepare_dataset(data_dir)

            if len(X) == 0 or len(y) == 0:
                self.root.after(0, lambda: messagebox.showerror("Error", 
                                                              "No training data found"))
                self.root.after(0, lambda: self.status_var.set("Error: No training data"))
                return

            # Train model
            accuracy, report, conf_matrix, class_names, f1, precision, recall, f1_avg, precision_avg, recall_avg = self.classifier.train(X, y)

            # Display results
            self.root.after(0, lambda: self._show_training_results(
                accuracy, report, conf_matrix, class_names, f1, precision, recall, f1_avg, precision_avg, recall_avg
            ))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                          f"Training error: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error training model"))

    def _show_training_results(self, accuracy, report, conf_matrix, class_names, f1, precision, recall, f1_avg, precision_avg, recall_avg):
        result_window = tk.Toplevel(self.root)
        result_window.title("Training Results")
        result_window.geometry("800x600")

        # Create a notebook (tabbed interface)
        notebook = ttk.Notebook(result_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Summary tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")

        # Display accuracy and average metrics
        metrics_frame = ttk.LabelFrame(summary_frame, text="Overall Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(metrics_frame, text=f"Accuracy: {accuracy:.4f}", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Label(metrics_frame, text=f"F1 Score (weighted): {f1_avg:.4f}", 
                 font=("Arial", 12)).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Label(metrics_frame, text=f"Precision (weighted): {precision_avg:.4f}", 
                 font=("Arial", 12)).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Label(metrics_frame, text=f"Recall (weighted): {recall_avg:.4f}", 
                 font=("Arial", 12)).pack(anchor=tk.W, padx=10, pady=5)

        # Display confusion matrix
        cm_frame = ttk.LabelFrame(summary_frame, text="Confusion Matrix")
        cm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a figure for the confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        im = ax_cm.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax_cm.set_title("Confusion Matrix")

        # Add colorbar
        plt.colorbar(im)

        # Add class labels
        tick_marks = np.arange(len(class_names))
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
        ax_cm.set_yticklabels(class_names)

        # Add text annotations to the confusion matrix
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax_cm.text(j, i, format(conf_matrix[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if conf_matrix[i, j] > thresh else "black")

        ax_cm.set_ylabel('True label')
        ax_cm.set_xlabel('Predicted label')
        plt.tight_layout()

        # Embed the confusion matrix plot in the UI
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Class metrics tab
        class_metrics_frame = ttk.Frame(notebook)
        notebook.add(class_metrics_frame, text="Class Metrics")

        # Create a table for class metrics
        metrics_table = ttk.Treeview(class_metrics_frame, columns=("Class", "F1", "Precision", "Recall"), show="headings")
        metrics_table.heading("Class", text="Class")
        metrics_table.heading("F1", text="F1 Score")
        metrics_table.heading("Precision", text="Precision")
        metrics_table.heading("Recall", text="Recall")

        metrics_table.column("Class", width=150)
        metrics_table.column("F1", width=100)
        metrics_table.column("Precision", width=100)
        metrics_table.column("Recall", width=100)

        for i, class_name in enumerate(class_names):
            metrics_table.insert("", "end", values=(
                class_name, 
                f"{f1[i]:.4f}", 
                f"{precision[i]:.4f}", 
                f"{recall[i]:.4f}"
            ))

        metrics_table.pack(fill=tk.X, padx=10, pady=10)

        # Create a chart to visualize class metrics
        chart_frame = ttk.LabelFrame(class_metrics_frame, text="Class Metrics Visualization")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a figure for the class metrics chart
        fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))

        # Set up the bar chart
        x = np.arange(len(class_names))  # the label locations
        width = 0.25  # the width of the bars

        # Create bars for each metric
        bars1 = ax_metrics.bar(x - width, f1, width, label='F1 Score')
        bars2 = ax_metrics.bar(x, precision, width, label='Precision')
        bars3 = ax_metrics.bar(x + width, recall, width, label='Recall')

        # Add labels, title and legend
        ax_metrics.set_xlabel('Classes')
        ax_metrics.set_ylabel('Scores')
        ax_metrics.set_title('Class Metrics Comparison')
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(class_names, rotation=45, ha="right")
        ax_metrics.set_ylim(0, 1.1)  # Metrics are between 0 and 1
        ax_metrics.legend()

        # Add value labels on top of bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax_metrics.annotate(f'{height:.2f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom', rotation=0, fontsize=8)

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

        plt.tight_layout()

        # Embed the chart in the UI
        canvas_metrics = FigureCanvasTkAgg(fig_metrics, master=chart_frame)
        canvas_metrics.draw()
        canvas_metrics.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Report tab
        report_frame = ttk.Frame(notebook)
        notebook.add(report_frame, text="Full Report")

        report_text = tk.Text(report_frame, wrap=tk.WORD, width=80, height=30)
        report_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        report_text.insert(tk.END, report)
        report_text.config(state=tk.DISABLED)

        self.status_var.set(f"Model trained with accuracy {accuracy:.4f}")

    def save_model(self):
        if not self.classifier.is_trained:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.classifier.save_model(file_path)
                self.status_var.set(f"Model saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save model: {str(e)}")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.classifier.load_model(file_path)
                self.status_var.set(f"Model loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load model: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseClassifierApp(root)
    root.mainloop()
