#pragma once

#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>

class VideoFaceDetector
{
public:
	VideoFaceDetector(const std::string cascadeFilePath, cv::VideoCapture &videoCapture);
	~VideoFaceDetector();

	cv::Point				getFrameAndDetect(cv::Mat &frame);
	cv::Point				operator>>(cv::Mat &frame);
	void					setVideoCapture(cv::VideoCapture &videoCapture);
	cv::VideoCapture		*videoCapture();
	void					setFaceCascade(const std::string cascadeFilePath);
	cv::CascadeClassifier	*faceCascade();
	cv::Rect				face();
	cv::Point				facePosition();
	void					setTemplateMatchingMaxDuration(double s);
	double					templateMatchingMaxDuration();

private:
	static const double		TICK_FREQUENCY;

	cv::VideoCapture		*m_videoCapture = NULL;
	cv::CascadeClassifier	*m_faceCascade = NULL;
	std::vector<cv::Rect>	m_allFaces;
	cv::Rect				m_trackedFace;
	cv::Rect				m_faceRoi;
	cv::Mat					m_faceTemplate;
	cv::Mat					m_matchingResult;
	bool					m_templateMatchingRunning = false;
	int64					m_templateMatchingStartTime = 0;
	int64					m_templateMatchingCurrentTime = 0;
	bool					m_foundFace = false;
	double					m_scale;
	cv::Point				m_facePosition;
	double					m_templateMatchingMaxDuration = 3;

	cv::Rect	doubleRectSize(cv::Rect &inputRect, cv::Rect &frameSize);
	cv::Rect	biggestFace(std::vector<cv::Rect> &faces);
	cv::Point	centerOfRect(cv::Rect rect);
	cv::Mat		getFaceTemplate(cv::Mat &frame, cv::Rect face);
	void		detectFaceAllSizes(cv::Mat &frame);
	void		detectFaceAroundRoi(cv::Mat &frame);
	void		detectFacesTemplateMatching(cv::Mat &frame);
};

