import os
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Load the model
model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}
reverse_labels_dict = {v: k for k, v in labels_dict.items()}


text_bg_path = "/Users/anirvanranjan/Downloads/fuck_dev/text_bg.png"
text_bg = Image.open(text_bg_path).convert("RGBA")
text_bg = np.array(text_bg)
text_bg = cv2.cvtColor(text_bg, cv2.COLOR_RGBA2BGRA)


def extract_features(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))
    if len(data_aux) < 84:
        data_aux.extend([0] * (84 - len(data_aux)))
    elif len(data_aux) > 84:
        data_aux = data_aux[:84]
    return data_aux


# Load letter images
letter_images = {}
letter_size = (300, 300)  # Updated size for letter images
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    img_path = f"/Users/anirvanranjan/Downloads/fuck_dev/alphabets/{letter}.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:  # If the image has an alpha channel
            img = cv2.resize(img, letter_size)
        else:
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA), letter_size)
            img = np.dstack([img, np.full(letter_size, 255, dtype=np.uint8)])
        letter_images[letter] = img


def draw_circular_progress(img, center, radius, progress, color):
    overlay = img.copy()
    cv2.circle(overlay, center, radius + 10, (255, 255, 255), -1)  # White background
    cv2.ellipse(
        overlay,
        center,
        (radius + 10, radius + 10),
        -90,
        0,
        360 * progress,
        color,
        thickness=-1,
    )  # Colored progress
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)


additional_options = ["DELETE", "SPACE", "LOCK", "SPEAK"]

# Load additional option images
option_size = (200, 200)  # Size for additional options
for option in additional_options:
    img_path = f"/Users/anirvanranjan/Downloads/fuck_dev/options/{option.lower()}.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:  # If the image has an alpha channel
            img = cv2.resize(img, option_size)
        else:
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA), option_size)
            img = np.dstack([img, np.full(option_size, 255, dtype=np.uint8)])
        letter_images[option] = img


def speak_text(text):
    os.system(f"say '{text}'")


current_text = ""
capture_mode = False
selected_letter = None
last_selected_letter = None
top_predictions = []
current_prediction = None
hover_start_time = {}


def perform_action(action):
    global current_text
    if action == "DELETE":
        current_text = current_text[:-1]
    elif action == "SPACE":
        current_text += " "
    elif action == "SPEAK":
        speak_text(current_text)
    elif action not in additional_options:
        current_text += action


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Try restarting the camera.")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and not capture_mode:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            data_aux = extract_features(hand_landmarks)

            try:
                prediction = model.predict([np.asarray(data_aux)])
                probabilities = model.predict_proba([np.asarray(data_aux)])[0]

                if isinstance(prediction[0], str):
                    predicted_character = prediction[0]
                else:
                    predicted_character = labels_dict[int(prediction[0])]

                current_prediction = predicted_character

                x1 = (
                    int(min(landmark.x for landmark in hand_landmarks.landmark) * W)
                    - 10
                )
                y1 = (
                    int(min(landmark.y for landmark in hand_landmarks.landmark) * H)
                    - 10
                )
                x2 = (
                    int(max(landmark.x for landmark in hand_landmarks.landmark) * W)
                    + 10
                )
                y2 = (
                    int(max(landmark.y for landmark in hand_landmarks.landmark) * H)
                    + 10
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(
                    frame,
                    predicted_character,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (252, 3, 136),
                    3,
                    cv2.LINE_AA,
                )

                # Get top 3 predictions
                top_3_indices = np.argsort(probabilities)[-5:][::-1]
                top_predictions = [
                    labels_dict[i] for i in top_3_indices if len(labels_dict[i]) == 1
                ][:3]

            except Exception as e:
                print(f"Error during prediction: {e}")
    text_bg_height = 60  # Adjust this value based on your text_bg image height
    text_region = frame[H - text_bg_height : H, 0:W]

    # Resize the text background to match the width of the frame
    resized_text_bg = cv2.resize(text_bg, (W, text_bg_height))

    # Blend the text background with the frame
    alpha = resized_text_bg[:, :, 3] / 255.0
    for c in range(0, 3):
        text_region[:, :, c] = (
            alpha * resized_text_bg[:, :, c] + (1 - alpha) * text_region[:, :, c]
        )

    cv2.putText(
        frame,
        current_text,
        (20, H - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (118, 103, 246),
        2,
        cv2.LINE_AA,
    )

    if capture_mode:
        hand_position = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_x = int(hand_landmarks.landmark[9].x * W)
            hand_y = int(hand_landmarks.landmark[9].y * H)
            hand_position = (hand_x, hand_y)

        display_options = top_predictions + additional_options
        for i, option in enumerate(display_options):
            if option in letter_images:
                img = letter_images[option]
                h, w = img.shape[:2]
                if i < 3:  # Top row (predicted alphabets)
                    x = i * (letter_size[0] + 10) + (W - 3 * letter_size[0] - 20) // 2
                    y = 10
                else:  # Bottom row (additional options)
                    x = (i - 3) * (option_size[0] + 10) + (
                        W - 4 * option_size[0] - 30
                    ) // 2
                    y = letter_size[1] + 30

                # Draw circular progress behind the PNG
                center = (x + w // 2, y + h // 2)
                radius = min(w, h) // 2 - 10
                if hand_position:
                    if x < hand_position[0] < x + w and y < hand_position[1] < y + h:
                        if option not in hover_start_time:
                            hover_start_time[option] = time.time()

                        hover_duration = time.time() - hover_start_time[option]
                        progress = min(hover_duration / 1.0, 1.0)
                        draw_circular_progress(
                            frame, center, radius, progress, (0, 255, 0)
                        )

                        if hover_duration >= 1:
                            selected_letter = option
                            if option == "LOCK":
                                if last_selected_letter:
                                    perform_action(last_selected_letter)
                                capture_mode = False  # Exit capture mode
                            else:
                                last_selected_letter = (
                                    option  # Update last selected letter
                                )
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        if option in hover_start_time:
                            del hover_start_time[option]

                # Overlay the option image with alpha blending
                roi = frame[y : y + h, x : x + w]
                alpha = img[:, :, 3] / 255.0
                rgb = img[:, :, :3]

                for c in range(0, 3):
                    roi[:, :, c] = alpha * rgb[:, :, c] + (1 - alpha) * roi[:, :, c]

                frame[y : y + h, x : x + w] = roi

    cv2.imshow("ASL Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # Press 'q' to quit
        break
    elif key == ord("c"):  # Press 'c' to toggle capture mode
        capture_mode = not capture_mode
        if capture_mode:
            cv2.imwrite("captured_frame.jpg", frame)
            print("Frame captured and saved as 'captured_frame.jpg'")
            hover_start_time.clear()  # Reset hover times when entering capture mode
        else:
            print(f"Current text: {current_text}")
            selected_letter = None
            last_selected_letter = None
            hover_start_time.clear()

cap.release()
cv2.destroyAllWindows()
