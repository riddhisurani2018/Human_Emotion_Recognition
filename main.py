from imutils import face_utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import dlib, cv2, os, pickle, sys

#loading pre-trained model for finding facial landmarks
#read more here : https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def train_n_test():
    #loading the input data
    labels = []
    landmarks_l = []
    try:
        subdirs = [x[0] for x in os.walk('data')]
    except:
        print('"data" folder does not exists please put it into same directory as the code')
        sys.exit(2)
    print('Reading the data..')
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        if (len(files) > 0):
            for file in files:
                img = cv2.imread(subdir+'/'+file)
                rects = detector(img, 0)
                shape = predictor(img, rects[0])
                landmarks_o = face_utils.shape_to_np(shape)
                landmarks = []
                for x in landmarks_o:
                    landmarks.extend(x.tolist())
                    #print(landmarks)
                if subdir.split('\\')[-1]=='afraid':
                    landmarks_l.append(landmarks)
                    labels.append('afraid')
                elif subdir.split('\\')[-1]=='angry':
                    landmarks_l.append(landmarks)
                    labels.append('angry')
                elif subdir.split('\\')[-1]=='happy':
                    landmarks_l.append(landmarks)
                    labels.append('happy')
                elif subdir.split('\\')[-1]=='sad':
                    landmarks_l.append(landmarks)
                    labels.append('sad')

    #normalizing the input data
    # : https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029
    landmarks = np.array(landmarks_l)
    mean = np.mean(landmarks,axis=(0,1))
    std = np.std(landmarks, axis=(0,1))
    landmarks = (landmarks-mean)/(std+1e-7)
    #print(landmarks)

    #splitting data into train and test set
    #Intro : https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
    #Link to the library page : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    landmarks_train,landmarks_test,labels_train,labels_test = train_test_split(landmarks,labels,test_size=.1,stratify=labels)

    #Building Classifier
    #Intro : https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72
    #Link to the library page : https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    print('Training the classifier..')
    svm = LinearSVC()
    svm.fit(landmarks_train,labels_train)
    pickle.dump({'model':svm,'mean':mean,'std':std}, open('saved_model.pkl','wb'))
    labels_pred = svm.predict(landmarks_test)
    
    #printing the results of model on test set
    #F1-Score : https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    #classification_report : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    print(classification_report(labels_test,labels_pred))
    print('Accuracy of trained model on testing data is :',accuracy_score(labels_test,labels_pred))

def test(landmarks):
    try:
        dct = pickle.load(open('saved_model.pkl','rb'))
        svm = dct['model']
        mean = dct['mean']
        std = dct['std']
        return svm.predict((np.array([landmarks])-mean)/(std+1e-7))
    except:
        print('Model is not properly saved, Please run filename.py train_n_test.')
    sys.exit(1)


#just some code for building a simple command line interface
hm = 'Run:\n"filename.py train_n_test" : for training and testing the model\n"filename.py test image_path" : for checking emotion of person in image'
if len(sys.argv)>1:
    if sys.argv[1] == 'train_n_test':
        train_n_test()
    elif sys.argv[1] == 'test':
        if len(sys.argv)>2:
            exists = os.path.isfile(sys.argv[2])
            if exists:
                img = cv2.imread(sys.argv[2])
                rects = detector(img, 0)
                shape = predictor(img, rects[0])
                landmarks_o = face_utils.shape_to_np(shape)
                landmarks = []
                for x in landmarks_o:
                    landmarks.extend(x.tolist())
                print('Emotion for the image is : '+test(landmarks)[0])
            else:
                print('File does not exists!')
        else:
            print('Wrong Syntax!\n'+hm)
    else:
        print('Wrong Syntax!\n'+hm)
else:
    print('Wrong Syntax!\n'+hm)
