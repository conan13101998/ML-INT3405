import face_recognition
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import os
import pymysql

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


def init_database():
    try:
        # cnx = mysql.connector.connect(user='root', password='iotlab2018',
        #                               host='sv-procon.uet.vnu.edu.vn',
        #                               database='age_gender_prediction')
        cnx = pymysql.connect(user='root', password='23081998',
                              host='127.0.0.1',
                              db='age_gender_prediction',
                              cursorclass=pymysql.cursors.DictCursor)

        return cnx
    except Exception as e:
        print("exception", str(e))
        return


def get_embedded_face():
    mycursor = conn.cursor()
    query = 'SELECT * FROM User_info'
    mycursor.execute(query)
    user_list = mycursor.fetchall()
    known_face_encodings = []
    known_face_names = []
    for row in user_list:
        usr_name = 'Name ' + str(row["user_id"])
        known_face_names.append(usr_name)
        known_face_encodings.append(np.frombuffer(row["face_embedding"]))
    mycursor.close()
    return known_face_encodings, known_face_names


def insert_user(face_embedding, age, gender):
    mycursor = conn.cursor()
    query = "INSERT INTO User_info (face_embedding, age, gender) VALUES (%s, %s, %s)"
    val = (face_embedding, age, gender)
    affected_cnt = mycursor.execute(query, val)
    conn.commit()
    mycursor.close()
    print(f"Inserted {affected_cnt}")
    return mycursor.lastrowid


def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


def ageModel():
    model = loadVggFaceModel()

    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)

    age_model.load_weights("model_checkpoint/age_model_weights.h5")

    return age_model


def genderModel():
    model = loadVggFaceModel()

    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    gender_model.load_weights("model_checkpoint/gender_model_weights.h5")

    return gender_model


conn = init_database()

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)

    known_face_encodings, known_face_names = get_embedded_face()
    similar_threshold = 0.5
    age_model = ageModel()
    gender_model = genderModel()
    output_indexes = np.array([i for i in range(0, 101)])

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    cnt = 0

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            try:
                for face_encoding in face_encodings:
                    if len(known_face_encodings) == 0:
                        print(f'Add new face with encoding: \n{face_encoding}')
                        known_face_encodings.append(face_encoding)
                        name = 'New ' + str(cnt)
                        known_face_names.append(name)
                        cnt += 1
                    else:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        # print(face_distances)
                        if face_distances[np.argmin(face_distances)] > similar_threshold:
                            print(f'Add new face with encoding: \n{face_encoding}')
                            known_face_encodings.append(face_encoding)
                            name = 'New ' + str(cnt)
                            known_face_names.append(name)
                            cnt += 1
                        else:
                            best_match_index = np.argmin(face_distances)
                            name = known_face_names[best_match_index]

                    face_names.append(name)
            except:
                pass

        process_this_frame = not process_this_frame

        for face_encoding, (top, right, bottom, left), name in zip(face_encodings, face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            w = abs(right - left)
            h = abs(top - bottom)
            gender = "M"
            apparent_age = 25

            try:
                margin = 40
                margin_x = int((w * margin) / 100)
                margin_y = int((h * margin) / 100)
                detected_face = frame[int(top - margin_y):int(top + h + margin_y),
                                int(left - margin_x):int(left + w + margin_x)]
            except:
                print("Detected face has no margin")

            try:
                # vgg-face expects inputs (224, 224, 3)
                detected_face = cv2.resize(detected_face, (224, 224))

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                # find out age and gender
                age_distributions = age_model.predict(img_pixels)
                apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))

                gender_distribution = gender_model.predict(img_pixels)[0]
                gender_index = np.argmax(gender_distribution)

                if gender_index == 0:
                    gender = "F"

                tmp = name.split()[0]
                idx = name.split()[1]
                if tmp == "New":
                    str_face_encoding = face_encoding.tostring()
                    name_idx = insert_user(str_face_encoding, int(apparent_age), int(gender_index))
                    known_face_names[known_face_names.index(name)] = 'Name ' + str(name_idx)
                    print("Changed name: ", name_idx)
            except Exception as e:
                # print("exception", str(e))
                pass

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            info = str(name) + ', ' + str(gender) + ', ' + str(apparent_age)
            cv2.putText(frame, info, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
