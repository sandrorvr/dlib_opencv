from threading import Thread
import cv2
import dlib
import os
import glob
import _pickle as cPickle
import numpy as np
import RPi.GPIO as gpio


def threadBoth(source=0):

    gpio.setmode(gpio.BCM)
    gpio.setup(5,gpio.OUT)
    video_getter = VideoGet(source).start()
    deteccao = processamento(video_getter.frame).start()
    video_shower = VideoShow(deteccao.nome,deteccao.face,video_getter.frame).start()
    
    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            deteccao.stop()
            break
        frame = video_getter.frame
        deteccao.imagem = frame
        video_shower.frame = frame
        video_shower.fila=deteccao.face
        video_shower.nome = deteccao.nome

    gpio.cleanup()



class VideoGet:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

class processamento:


    def __init__(self, imagem):
        self.imagem=imagem
        self.stopped = False
        self.face=None
        self.nome=None

    def start(self):
        Thread(target=self.detector, args=()).start()
        return self
#ESTA FUNÇÃO USA A TECNICA HAARCASCADE
    '''def detector(self):
        detector = dlib.get_frontal_face_detector()
        while not self.stopped:
            facesDetectadas = detector(self.imagem)
            self.face=facesDetectadas
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True'''


    def detector(self):
        detectorFace = dlib.get_frontal_face_detector()
        detectorPontos = dlib.shape_predictor("C:\\Users\\sandro\\Documents\\Python Scripts\\udemy\\recursos\\shape_predictor_68_face_landmarks.dat")
        reconhecimentoFacial = dlib.face_recognition_model_v1("C:\\Users\\sandro\\Documents\\Python Scripts\\udemy\\recursos\\dlib_face_recognition_resnet_model_v1.dat")
        indices = np.load("C:\\Users\\sandro\\Documents\\Python Scripts\\reconhecimento_face\\script_dlib\\indices_rn.pickle")
        descritoresFaciais = np.load("C:\\Users\\sandro\\Documents\\Python Scripts\\reconhecimento_face\\script_dlib\\descritores_rn.npy")
        limiar = 0.5
        classificador_face = cv2.CascadeClassifier('C:\\Users\\sandro\\Documents\\Python Scripts\\reconhecimento_face\\script_dlib\\face.xml')
        while not self.stopped:
            gray = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY)
            facesDetectadas = classificador_face.detectMultiScale(gray)
            self.face=facesDetectadas
            facesDetectadas = detectorFace(self.imagem)
            if Len(facesDetectadas)!=0:
                gpio.output(5,gpio.HIGH)
            else:
                gpio.output(5,gpio.LOW)
            self.nome = ' '
            for face in facesDetectadas:
                pontosFaciais = detectorPontos(self.imagem, face)
                descritorFacial = reconhecimentoFacial.compute_face_descriptor(self.imagem, pontosFaciais)
                listaDescritorFacial = [fd for fd in descritorFacial]
                npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
                npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
                distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
                minimo = np.argmin(distancias)
                distanciaMinima = distancias[minimo]
                if distanciaMinima <= limiar:
                    self.nome = os.path.split(indices[minimo])[1].split("_")[0]
                else:
                    self.nome = ' '
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True


    def stop(self):
            self.stopped = True


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, nome, fila, frame=None):
        self.frame = frame
        self.stopped = False
        self.fila = fila
        self.nome = nome

    def start(self):
        Thread(target=self.show, args=()).start()
        return self
#ESTA FUNCAO USA A TECNICA HOG
    '''def show(self):
        while not self.stopped:
            #print(self.fila)
            try:
                for face in self.fila:
                    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
                    cv2.rectangle(self.frame, (e, t), (d, b), (0, 255, 255), 2)
            except TypeError:
                print('DEU MERDA NEGAO')
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True'''

    def show(self):
        classificador_olho = cv2.CascadeClassifier('C:\\Users\\sandro\\Documents\\Python Scripts\\script_dlib\\olho.xml')
        while not self.stopped:
            #print(self.fila)
            '''try:
                for (x,y,l,h) in self.fila:
                    mascara=self.frame[y:y+h,x:x+l]
                    olho = classificador_olho.detectMultiScale(mascara)
                    if len(olho)!=0:
                        cv2.rectangle(self.frame,(x,y),(x+l,y+h),(255,0,0),2)
            except TypeError:
                print('BOTA A CARA NEGAO')'''
            cv2.putText(self.frame, self.nome, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),2,cv2.LINE_AA)
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True


    def stop(self):
        self.stopped = True






threadBoth(source=0)


