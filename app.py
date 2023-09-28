from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp

app = Flask(__name__, static_url_path='/static')

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    ih, iw, ic = img.shape
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    cof = 1

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        xc1, yc1 = int(landmarks.landmark[168].x * iw), int(landmarks.landmark[168].y * ih)
        xc2, yc2 = int(landmarks.landmark[6].x * iw), int(landmarks.landmark[6].y * ih)
        cof = 180 / (yc2 - yc1)

        x1, y1 = int(landmarks.landmark[94].x * iw), int(landmarks.landmark[94].y * ih)
        x2, y2 = int(landmarks.landmark[152].x * iw), int(landmarks.landmark[152].y * ih)

        x5, y5 = int(landmarks.landmark[112].x * iw), int(landmarks.landmark[112].y * ih)
        x6, y6 = int(landmarks.landmark[186].x * iw), int(landmarks.landmark[186].y * ih)

        if cof >= 1 and y2 - y1 > 1063:
            cof = (yc2 - yc1) / 180

    DVO_state = (y6 - y5) * cof * (0.059)
    return DVO_state

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_path = './images/' + file.filename
    file.save(image_path)

    DVO_state = extract_landmarks(image_path)

    DVO_threshold = 7
    if DVO_state - DVO_threshold <= 0:
        DVO_state = "Good DVO"
    else:
        DVO_state = "DVO is " + ("increased" if DVO_state > DVO_threshold else "decreased")

    results = {
        'DVO State': DVO_state,
        'Suggested Therapeutic DVO': DVO_state,
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
