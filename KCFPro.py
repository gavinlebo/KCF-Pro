import cv2
import pandas
from sklearn import linear_model
import numpy as np

class TrackerData:
    def __init__(self, video, waitTime):
        self.tracker = cv2.TrackerKCF_create()
        self.previousPredictions = []
        self.waitTime = waitTime
        self.video = video
        self.framenumber = 0
        
    def init(self, frame, box):
        self.tracker.init(frame, box)
        self.framenumber = 0

    def update(self, frame):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        timeelapsed = self.framenumber / fps
        projectedFinish = (self.waitTime/fps) + timeelapsed
        
        ret, bbox = self.tracker.update(frame)
        
        if ret:
            if len(self.previousPredictions) >= 10:
                del self.previousPredictions[0]
            self.previousPredictions.append((bbox[0], bbox[1], bbox[2], bbox[3], timeelapsed))
            self.framenumber += 1
            return ret, bbox
        else:
            #proccess data of results of all past frames into dataframe
            df = pandas.DataFrame(self.previousPredictions, dtype=float)
            Time = df[[4]]
            X = df[0]
            Y = df[1]
            
            #Make a linearregression model for both coordinates in relation to time, and fit corresponding data
            Xpredictor = linear_model.LinearRegression()
            Ypredictor = linear_model.LinearRegression()
            Xpredictor.fit(Time, X)
            Ypredictor.fit(Time, Y)

            #predict the coordinates at the projected time to finish, and create a new box with the results
            predictedX = np.round(Xpredictor.predict([[projectedFinish]]))
            predictedY = np.round(Ypredictor.predict([[projectedFinish]]))
            newbox = list((int(predictedX), int(predictedY), int(self.previousPredictions[-1][2] * 1.5), int(self.previousPredictions[-1][3] * 1.5)))

            for i in range(self.waitTime):
                ret, frame = self.video.read()
            
            #reinitialize tracker
            try:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame, newbox)
            except:
                print("ERROR: Backup Tracker Failed or video ended")
                return False, False
            #return coordinates of the new box
            
            return True, newbox

def TrackerKCFPro_create(video, waitTime):
    return TrackerData(video, waitTime)
