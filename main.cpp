#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>

#include <iostream>

const cv::Scalar	COLOR_BLUE(255., 0., 0.);
const cv::String	WINDOW_NAME("Camera video");
const cv::String	CASCADE_FILE("haarcascade_frontalface_default.xml");
const double		TICK_FREQUENCY = cv::getTickFrequency();

// Pointer to cascade classifier used to detect face
cv::CascadeClassifier *cascade_classifier;

// Vector used to store detected faces
std::vector<cv::Rect> faces;

// Measure time and fps
int64 start_time, end_time;
double average_FPS = 0;

// Rectangle around tracked face and ROI around it
cv::Rect face, face_roi;

// Matrix holding camera frame
cv::Mat frame, face_template, matching_result;

// Template matching failsafe
bool template_matching_running = false;
int64 template_matching_start_time = 0, template_matching_current_time = 0;

// Flag indicating if a face was found
bool found_face = false;

// Doubles the size of a rectangle and keeps it's center in the same place
// Makes sure the new rectange is inside given limits 
cv::Rect doubleRectSize(cv::Rect &input_rect, cv::Rect keep_inside)
{
    cv::Rect output_rect;
    // Double rect size
    output_rect.width	= input_rect.width * 2;
    output_rect.height	= input_rect.height * 2;

    // Center rect around original center
    output_rect.x = input_rect.x - input_rect.width / 2;
    output_rect.y = input_rect.y - input_rect.height / 2;

    // Handle edge cases
    if (output_rect.x < keep_inside.x) {
        output_rect.width += output_rect.x;
        output_rect.x = keep_inside.x;
    }
    if (output_rect.y < keep_inside.y) {
        output_rect.height += output_rect.y;
        output_rect.y = keep_inside.y;
    }
    
    if (output_rect.x + output_rect.width > keep_inside.width) {
        output_rect.width = keep_inside.width - output_rect.x;
    }
    if (output_rect.y + output_rect.height > keep_inside.height) {
        output_rect.height = keep_inside.height- output_rect.y;
    }

    return output_rect;
}

// Returns rectangle with biggest surface area in array of rects
cv::Rect biggestFace(std::vector<cv::Rect> &faces_arr) 
{
    int biggest = 0;
    for (int i = 0; i < faces_arr.size(); i++) {
        if (faces_arr[i].area() > faces_arr[biggest].area()) {
            biggest = i;
        }
    }
    return faces_arr[biggest];
}

// Start measuring time
void startMeasuringTime() 
{
    start_time = cv::getTickCount();
}

// Stop measuring time and output FPS
void stopMeasuringTime() 
{
    end_time = cv::getTickCount();
    double time_per_frame = (double)((end_time - start_time) / TICK_FREQUENCY);
    double curr_FPS = 1. / time_per_frame;
    average_FPS = (3 * average_FPS + curr_FPS) / 4;

    std::cout << "Average FPS = " << average_FPS << "\n";
}

void showFrame(cv::Mat &frame)
{
    cv::imshow(WINDOW_NAME, frame);
    if (cv::waitKey(25) == 27) exit(0);
}

void drawRectAroundFace(cv::Mat &frame)
{
    cv::rectangle(frame, face, COLOR_BLUE);
}

void detectFaceAllSizes(cv::Mat &frame)
{
    // Minimum face size is 1/5th of screen height
    // Maximum face size is 2/3rds of screen height
    cascade_classifier->detectMultiScale(frame, faces, 1.1, 3, 0,
        cv::Size(frame.rows / 5, frame.rows / 5),
        cv::Size(frame.rows * 2 / 3, frame.rows * 2 / 3));

    if (faces.empty()) return;

    found_face = true;

    // Locate biggest face;
    face = biggestFace(faces); 

    // Copy face template
    face_template = frame(face).clone(); 
    
    // Calculate roi
    face_roi = doubleRectSize(face, cv::Rect(0, 0, frame.cols, frame.rows)); 
}

void detectFaceAroundRoi(cv::Mat &frame)
{
    // Detect faces sized +/-20% off biggest face in previous search
    cascade_classifier->detectMultiScale(frame(face_roi), faces, 1.1, 3, 0,
        cv::Size(face.width * 8 / 10, face.height * 8 / 10),
        cv::Size(face.width * 12 / 10, face.width * 12 / 10));

    if (faces.empty())
    {
        // Activate template matching if not already started and start timer
        template_matching_running = true;
        if (template_matching_start_time == 0)
            template_matching_start_time = cv::getTickCount();
        return;
    }

    // Turn off template matching if running and reset timer
    template_matching_running = false;
    template_matching_current_time = template_matching_start_time = 0;

    // Get detected face
    face = biggestFace(faces); 

    // Add roi offset to face
    face.x += face_roi.x; 
    face.y += face_roi.y;

    // Get face template
    face_template = frame(face).clone(); 

    // Calculate roi
    face_roi = doubleRectSize(face, cv::Rect(0, 0, frame.cols, frame.rows));
}

void detectFacesTemplateMatching(cv::Mat &frame)
{
    // Calculate duration of template matching
    template_matching_current_time = cv::getTickCount();
    double duration = (double)(template_matching_current_time - template_matching_start_time) / TICK_FREQUENCY;

    // If template matching lasts for more than 2 seconds face is possibly lost
    // so disable it and redetect using cascades
    if (duration > 2) {
        found_face = false;
        template_matching_running = false;
        template_matching_start_time = template_matching_current_time = 0;
    }

    // Template matching with last known face 
    cv::matchTemplate(frame(face_roi), face_template, matching_result, CV_TM_CCOEFF);
    cv::normalize(matching_result, matching_result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(matching_result, &min, &max, &min_loc, &max_loc);

    // Add roi offset to face position
    max_loc.x += face_roi.x;
    max_loc.y += face_roi.y;

    // Get detected face
    face = cv::Rect(max_loc.x, max_loc.y, face.width, face.height);

    // Get new face template
    face_template = frame(face).clone();

    // Calculate face roi
    face_roi = doubleRectSize(face, cv::Rect(0, 0, frame.cols, frame.rows));
}


int main(int argc, char** argv)
{
    // Try opening camera
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cout << "Error getting camera...\n";
        exit(1);
    }

    // Create cascade classifier
    cascade_classifier = new cv::CascadeClassifier(CASCADE_FILE);
    if (cascade_classifier->empty()) {
        std::cout << "Error creating cascade classifier. Make sure the file \n"
            "\t" << CASCADE_FILE << "\n"
            "is in working directory.\n";
        exit(1);
    }

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO|cv::WINDOW_AUTOSIZE);

    // Find initial face on screen...
    while (! found_face) 
    {
        startMeasuringTime();
        camera >> frame; 
        detectFaceAllSizes(frame); // Detect using cascades over whole image
        drawRectAroundFace(frame); 
        showFrame(frame); 
        stopMeasuringTime();
        
        // Once the face is found...
        while (found_face)
        {
            startMeasuringTime();
            camera >> frame;
            detectFaceAroundRoi(frame); // Detect using cascades only in ROI
            if (template_matching_running) { // If Haar detection failed...
                detectFacesTemplateMatching(frame); // Detect using template matching
            }
            drawRectAroundFace(frame);
            showFrame(frame);
            stopMeasuringTime();
        }
    }
    delete cascade_classifier;
    return 0;
}