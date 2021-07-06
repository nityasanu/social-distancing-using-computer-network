import time

# import the necessary packages
import cv2
import numpy as np
import pymongo

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONFIDENCE = 0.5
thresh = 0.5

# Create and connect to the mongo instance
# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# mydb = myclient["mldataset"]
# mycol = mydb["areacode"]


# this lets us define the minimum safe distance (in pixels) that two people can be
# from each other
def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0

np.random.seed(42)

weightsPath = "./yolov3.weights"
configPath = "./yolov3.cfg"
labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture('./videos/town.mp4')
videoWriter = None
(W, H) = (None, None)

fl = 0
q = 0
while True:

    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        # grab the dimensions of the frame
        (H, W) = frame.shape[:2]
        q = W

    # Need to set the screen resolution here
    frame = frame[1:650, 100:q]
    (H, W) = frame.shape[:2]
    # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    # Forward pass
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:
            # extract the class ID and confidence (i.e., probability)
			# of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
            if LABELS[classID] == "person":
                if confidence > MIN_CONFIDENCE:
                
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, thresh)

    if len(idxs) > 0:

        status = list()
        idf = idxs.flatten()
        close_pair = list()
        s_close_pair = list()
        center = list()
        dist = list()
        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])

            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(center[i], center[j])

                if g == 1:

                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2

        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0

        for i in idf:

            sub_img = frame[10:170, 10:W - 10]
            black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

            res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.01, 1.0)

            frame[10:170, 10:W - 10] = res

            cv2.putText(frame, "TARP Project Group 3 - Social Distancing", (210, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.rectangle(frame, (20, 60), (510, 160), (170, 170, 170), 2)
            cv2.putText(frame, "Group members: ", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "17BCE2374", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "17BCE2328", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "17BCE2188", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # cv2.rectangle(frame, (535, 60), (W - 20, 160), (170, 170, 170), 2)
            # cv2.putText(frame, "Bounding box shows the level of risk to the person.", (545, 80),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "17BCI0198", (565, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
            cv2.putText(frame, "17BCE0067", (565, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
            cv2.putText(frame, "17BCE2349", (565, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            tot_str = "TOTAL COUNT: " + str(total_p)
            high_str = "HIGH RISK COUNT: " + str(high_risk_p)
            low_str = "LOW RISK COUNT: " + str(low_risk_p)
            safe_str = "SAFE COUNT: " + str(safe_p)

            # For offline demonstration only
            # Writing to CSV file
            with open('dataset.csv','a') as file:
                file.write(str(start) + "," + str(end)+ "," + str(total_p)+ "," + str(high_risk_p)+ "," + str(low_risk_p)+ "," + str(safe_p))
                file.write('\n')

            ## What to do on production?
            # if True:
            #     mycol.insert_one({
            #         "start":str(start),
            #         "end":str(end),
            #         "total_p":str(total_p),
            #         "high_risk_p":str(high_risk_p),
            #         "low_risk_p":str(low_risk_p),                    
            #         "safe_p":str(safe_p),
            #     })

            sub_img = frame[H - 120:H, 0:210]
            black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

            res = cv2.addWeighted(sub_img, 0.8, black_rect, 0.2, 1.0)

            frame[H - 120:H, 0:210] = res

            cv2.putText(frame, tot_str, (10, H - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, safe_str, (10, H - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, low_str, (10, H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 1)
            cv2.putText(frame, high_str, (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

        cv2.imshow('Social distancing analyser', frame)
        cv2.waitKey(1)

    if videoWriter is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = cv2.VideoWriter("output.mp4", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    videoWriter.write(frame)
print("Processing finished: open output.mp4")
videoWriter.release()
vs.release()
