import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import mediapipe as mp
import random
from caption_model import CaptionGenerationModel

# Define preprocessing parameters
target_width = 224  # Adjust based on your model's input size
target_height = 224

# Frame rate sampling parameters
sampling_rate = 5  # Sample every 5th frame (adjust as needed)

# Data augmentation parameters (adjust probabilities as needed)
augmentation_prob = 0.5  # Probability of applying augmentation
crop_prob = 0.3  # Probability of random cropping
flip_prob = 0.2  # Probability of horizontal flipping

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def preprocess_video(video_path):
    """
    Preprocesses a video for sign language captioning with frame rate sampling
    and optional data augmentation.

    Args:
        video_path: Path to the video file.

    Returns:
        A list of preprocessed frames.
    """
    cap = cv2.VideoCapture(video_path)
    preprocessed_frames = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Sample frames at a consistent rate
        if frame_count % sampling_rate == 0:
            # Resize frame
            frame = cv2.resize(frame, (target_width, target_height))

            # Convert to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Normalize pixel values (0-1)
            frame = frame.astype('float32') / 255.0

            # Apply data augmentation with a certain probability
            if random.random() < augmentation_prob:
                # Random cropping
                if random.random() < crop_prob:
                    y0, x0 = random.randint(0, frame.shape[0] - target_height), random.randint(0, frame.shape[1] - target_width)
                    frame = frame[y0:y0 + target_height, x0:x0 + target_width]

                # Horizontal flipping
                if random.random() < flip_prob:
                    frame = cv2.flip(frame, 1)  # Flip horizontally

            # Detect hands using MediaPipe (optional)
            results = mp_hands.process(frame)

            # Extract hand landmarks (optional)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Process landmarks here (e.g., calculate bounding box)
                    pass

            preprocessed_frames.append(frame)

        frame_count += 1

    cap.release()
    return preprocessed_frames

# Define path to your dataset folder
dataset_dir = "C:\\Users\\Hp\\Documents\\env\\archive"

# Tokenization function (replace with your actual tokenization logic)
def tokenize_caption(caption, word2idx):
    tokens = caption.lower().split()
    token_ids = [word2idx[word] if word in word2idx else word2idx["<UNK>"] for word in tokens]
    return token_ids

# Define vocabulary (replace with your actual vocabulary)
word2idx = {"hello": 0, "how": 1, "are": 2, "you": 3, "goodbye": 4, "<UNK>": 5, "<PAD>": 6}  # Example vocabulary
idx2word = {idx: word for word, idx in word2idx.items()}

class SignLanguageDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.label_to_caption = {}  # Dictionary mapping label (integer) to caption (string)

        # Load captions from CSV file (assuming 'label' and 'caption' are the correct column names)
        csv_file_path = os.path.join(dataset_dir, "sign_mnist_train.csv")
        df = pd.read_csv(csv_file_path)

        for index, row in df.iloc[1:].iterrows():
            label = row['label']
            caption = " ".join(str(pixel) for pixel in row[1:])
            self.label_to_caption[label] = caption

    def __len__(self):
        return len(self.label_to_caption)

    def __getitem__(self, idx):
        label = list(self.label_to_caption.keys())[idx]
        video_path = os.path.join(self.dataset_dir, f"{label}.mp4")
        caption = self.label_to_caption[label]

        # Preprocess video and extract features
        preprocessed_features = preprocess_video(video_path)

        # Convert list of preprocessed frames to tensor
        preprocessed_features = torch.tensor(preprocessed_features)

        # Tokenize caption
        caption = tokenize_caption(caption, word2idx)

        # Convert caption to tensor with padding
        caption = torch.tensor(caption, dtype=torch.long)

        return preprocessed_features, caption


# Training function
def train(model, train_loader, optimizer, criterion, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        for features, captions in train_loader:
            features = features.to(device)
            captions = captions.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features, captions[:, :-1])  # Exclude last word for teacher forcing
            loss = criterion(outputs.transpose(1, 2), captions[:, 1:])  # Calculate loss on shifted captions (teacher forcing)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

        # Print training progress
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


# Function to generate caption for a new video
def generate_caption(model, video_features, word2idx, idx2word, max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Convert features to a batch of size 1
    video_features = video_features.unsqueeze(0).to(device)

    # Start with the start token (e.g., "<SOS>")
    start_token = word2idx["<SOS>"]
    caption = torch.tensor([start_token], dtype=torch.long).to(device)

    # Generate caption one word at a time
    for i in range(max_len):
        predictions = model(video_features, caption[:-1])  # Exclude last word
        predicted_word_idx = torch.argmax(predictions[:, -1, :], dim=1).item()  # Get the most likely word

        # Check for end token or max length reached
        if predicted_word_idx == word2idx["<EOS>"] or i + 1 == max_len:
            break

        caption = torch.cat((caption, torch.tensor([predicted_word_idx], dtype=torch.long).to(device)), dim=0)

    # Convert word indices to actual words
    caption = caption.cpu().numpy()[1:]  # Remove start token
    caption = [idx2word[idx] for idx in caption]

    return caption


# Main program
if __name__ == "__main__":
    # Load your dataset
    dataset = SignLanguageDataset(dataset_dir)

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Define model, loss function, and optimizer
    model = CaptionGenerationModel(len(word2idx), embedding_dim=256, hidden_dim=512)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, optimizer, criterion)

    # Example usage of generating caption for a new video
    # Assuming you have a preprocessed video features tensor 'video_features'
    video_features = torch.randn(10, 3, target_height, target_width)  # Example tensor shape (num_frames, channels, height, width)
    caption = generate_caption(model, video_features, word2idx, idx2word)
    print("Generated Caption:", " ".join(caption))
