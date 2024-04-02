#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const String model = "opencv_face_detector_uint8.pb";                  //텐서플로우 사용
const String config = "opencv_face_detector.pbtxt";                    //텐서플로우 사용

int main(void)
{
	VideoCapture cap(0);                                               //카메라 열기

	if (!cap.isOpened()) {                                             //카메라가 열리지 않으면 fail출력 및 종료
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	Net net = readNet(model, config);                                  //model, config 파일을 이용하여 Net net 객체  생성
	if (net.empty()) {                                                 //net 객체 생성 실패 시 fail출력 및 종료
		cerr << "Net open failed!" << endl;
		return -1;
	}


	Mat frame, faceROI;
	while (true) {           
		cap >> frame;                                                  //카메라의 매 프레임을 frame 변수에 저장하고
		if (frame.empty())                                             //frame을 받아오지 못하면 종료
			break;

		// 모폴로지 실행 //
		Mat gray = frame.clone();                                      //frame 복제하여 gray 변수로 저장
		cvtColor(gray, gray, COLOR_BGR2GRAY);                          //gray 변수를 그레이스케일로 변환
		Mat bin, dst;
		threshold(gray, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);     //오츠 알고리즘으로 임계값 자동으로 설정하고 자동 이진화 수행하여 bin 변수에 저장
		morphologyEx(bin, dst, MORPH_OPEN, Mat());                     //bin 영상의 모폴로지 열기(축소->팽창)을 실행하여 dst 변수에 저장
		// 모폴로지 실행 //

		Mat blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123));     //frame 영상으로부터 블롭(영상 등의 데이터를 포함할 수 있는 다차원 데이터 표현 방식) 생성
		net.setInput(blob);                                                            //입력 영상에 1을 곱하고, 출력 영상 사이즈는 300x300, RGB 순서로 scalar 지정
		Mat res = net.forward();                                                       //네트워크 실행 결과를 res에 저장

		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>());

		for (int i = 0; i < detect.rows; i++) { 
			float confidence = detect.at<float>(i, 2);
			if (confidence < 0.5)                                     //결과 행렬의 신뢰되 값이 0.5보다 작으면 무시
				break;

			int x1 = cvRound(detect.at<float>(i, 3) * frame.cols);
			int y1 = cvRound(detect.at<float>(i, 4) * frame.rows);
			int x2 = cvRound(detect.at<float>(i, 5) * frame.cols);
			int y2 = cvRound(detect.at<float>(i, 6) * frame.rows);    //얼굴 검출 사각형 영역의 좌표 계산(좌측상단(x1,y1), 우측하단(x2,y2))

			
			rectangle(frame, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 255, 0));   //frame에서 얼굴 영역을 초록색으로 검출, 신뢰도 출력
			gray = gray(Rect(Point(x1, y1), Point(x2, y2)));                           //검출된 얼굴만 gray 변수에 저장
			
			String label = format("Face: %4.3f", confidence);
			putText(frame, label, Point(x1, y1 - 1), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0));
		}
		
		imshow("frame", frame);                                       //실시간 영상 출력 및 얼굴 검출
		imshow("morphology", gray);                                    //모폴로지 열기 영상도 동시에 출력

		if (waitKey(1) == 27)
			break;
	}
	return 0;
}

