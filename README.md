# [Youtube video](https://youtu.be/lkFBWUjwDl8)

# Usage

First you need to create a `VideoCapture` object that you'll use as a source. Then pass the path to your cascade file along with the `VideoCapture` object to the `VideoFaceDetector`.

    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        fprintf(stderr, "Error getting camera...\n");
        exit(1);
    }
    VideoFaceDetector detector(CASCADE_FILE, camera);
    
Now you can use your `VideoFaceDetector` object just like a regular `VideoCapture` object to get frames. It will automatically detect a face in the frame which you can access with `VideoFaceDetector::face()` and `VideoFaceDetector::facePosition()`

    cv::Mat frame;
    detector >> frame; // same as detector.getFrameAndDetect(frame);
    
    cv::Rect face = detector.face();
    cv::Point facePos = detector.facePosition();
    
You can change the `VideoCapture` object the detector is hooked to with `VideoFaceDetector::setVideoCapture(cv::VideoCapture &videoCapture)` and retrieve it with `VideoFaceDetector::videoCapture()`.

You can change the cascade file with `VideoFaceDetector::setFaceCascade(const std::string cascadeFilePath)` and retrieve the cascade classifier with `VideoFaceDetector::faceCascade()`.

You can change the size to which the detector resizes the frames internally with `VideoFaceDetector::setResizedWidth()` and retrieve it with `VideoFaceDetector::resizedWidth()`. This can speed up the detection but the tradeoff is precision. The default setting is 320px.

You can change the template matching max duration with `VideoFaceDetector::setTemplateMatchingMaxDuration(const double s)` and retrieve it with `VideoFaceDetector::templateMatchingMaxDuration()`. The default value is 3 seconds. This is the max time the algorithm tracks using template matching and after this time the algorithm starts tracking in the whole image again. See algorithm description for more details.
 
# Head detection and real time tracking

I've recently been woriking on a project which required head tracking. It ran on Android so it had to be efficient in order to run fast and smooth on mobile devices.
    
I tried to find a reliable and fast face tracking algorithm online but all I found was slow-ih implementations so I decided to make my own algorithm that would run at least 15 fps on mobile devices. I only needed to track the primary user so I used that fact to speed up the algorithm.
    
# Haar cascades 
    
Haar cascades are currently the fastest face detection algorithm we have. However, the algorithm needs some fine tuning to get really fast and it has one flaw. If the face is at an angle it can't detect it.

# Template matching
        
Template matching is a technique used to find a smaller image in a larger one. It works by sliding the small image accross the big one and it uses math to calculate which part of the bigger image is most likely to be the small image. This algorithm is nice because it always returns a value, unlike Haar cascades which is returns a position only if it finds a face.
    
# The algorithm
        
The algorithm I came up with is a hybrid using Haar cascades with template matching as a fallback when Haar fails. It has two routines for detecting faces using Haar cascades. One routine is used for the inital detection of a face and it's slow and clunky. Once it's found, it's position is remembered and a region of interest is calculated around it. The other routine is used to detect faces in this region of interest. This speeds up the detection significantly. Also it searches for a face +/-20% size of the face from the prior frame. This also boosts the performance of the algorithm.

This makes the algorithm fast but it's still shitty as it fails when you rotate your face at an angle. Template matching to the rescue. If Haar cascades fail, the template matching algorithm calculates the most likely position of face based on the last detected face template. This makes the algorithm reliable and tracks the face pretty good. Template matching continues until one of two things happen. Either Haar cascades redetect a face, or the template matching fails and the tracking windows loses the face. In the face that template matching fails, there is a timer implemented that will turn off template matching after 2 seconds of tracking and reinitialize face tracking with the slower Haar cascades over the complete frame.
    
I'm not sure any of this made any sense to you so here's this diagram that will make it all really easy to understand... I hope.
 
[![Algorithm diagram](https://raw.githubusercontent.com/mc-jesus/face_detect_n_track/master/image/img.png)](https://youtu.be/lkFBWUjwDl8)
