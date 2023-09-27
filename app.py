from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    # Handle image upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image to a temporary location
    image_path = './images/' + file.filename
    file.save(image_path)

    img = cv2.imread(image_path)

    cof = 1
    # mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        ih, iw, ic = img.shape

        # cof
        xc1, yc1 = int(landmarks.landmark[168].x *
                       iw), int(landmarks.landmark[168].y*ih)

        xc2, yc2 = int(landmarks.landmark[6].x *
                       iw), int(landmarks.landmark[6].y*ih)
        # cv2.line(img, (xc1, yc1), (xc2, yc2), (0, 255, 0), 4)
        cof = 180/(yc2-yc1)

        # DVO
        x1, y1 = int(landmarks.landmark[94].x *
                     iw), int(landmarks.landmark[94].y*ih)

        x2, y2 = int(landmarks.landmark[152].x *
                     iw), int(landmarks.landmark[152].y*ih)
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # line bi pulpaire
        # x3, y3 = int(landmarks.landmark[158].x *
        #              iw), int(landmarks.landmark[158].y*ih)

        # x4, y4 = int(landmarks.landmark[385].x *
        #              iw), int(landmarks.landmark[385].y*ih)
        # cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 4)

        # line angle interne de l’œil – commissure labiale
        x5, y5 = int(landmarks.landmark[112].x *
                     iw), int(landmarks.landmark[112].y*ih)

        x6, y6 = int(landmarks.landmark[186].x *
                     iw), int(landmarks.landmark[186].y*ih)
        # cv2.line(img, (x5, y5), (x6, y6), (0, 255, 0), 4)
        if cof >= 1 and y2-y1 > 1063:
            cof = (yc2-yc1)/180
        # print(f"cof is equal to: {cof}")
        # print(f"DVO in pixels is equal to: {(y2-y1)*cof:.2f} px")
        # print(
        #     f"line (angle interne de l’œil – commissure labiale) in pixels is equal to: {(y6-y5)*cof:.2f} px")
        # print(f"line bi puplaire in pixels is equal to: {x4-x3}")
        # print(f"==========the measurements in mm========")
        # print(f"DVO in mm is equal to: {(y2-y1)*cof*(0.059):.2f} mm")
        # print(
        #     f"line (angle interne de l’œil – commissure labiale) in mm is equal to: {(y6-y5)*cof*(0.059):.2f} mm")

    DVO_state = (y6-y5)*cof*(0.059)
    if DVO_state-7 <= (y2 - y1) * cof*(0.059) <= DVO_state+7:
        DVO_state = "Good DVO"
    elif (y2 - y1) * cof*(0.059) < DVO_state-7:
        DVO_state = "DVO is decreased"
    elif (y2 - y1) * cof*(0.059) > DVO_state+7:
        DVO_state = "DVO is increased"

    # Return the results as JSON
    results = {
        'DVO  State': DVO_state,
        'Suggested Therapeutic DVO': (y6 - y5) * cof*(0.059),
        # Add more measurements if needed...
    }

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=False)
