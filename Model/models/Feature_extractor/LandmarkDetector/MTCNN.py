from facenet_pytorch import InceptionResnetV1, MTCNN

def MTCNN_creation():

    landmarks_detector = MTCNN(keep_all=True)

    return landmarks_detector