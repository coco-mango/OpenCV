#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void blurring_gaussian();                    
void on_level_change(int pos, void* userdata);
void unsharp_mask();

Mat img, src, gaussian, unsharp;

int main(void)
{
	blurring_gaussian();                                                      //가우시안 필터 함수 호출
	unsharp_mask();                                                           //언샤프닝 필터 함수 호출
	return 0;
}
 
void blurring_gaussian()                                                      //가우시안 필터 함수
{
	src = imread("rose.bmp", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;                                       //이미지를 불려오지 못했을 경우 fail 출력
	}
	imshow("src", src);                                                             //불려온 이미지 출력

	int start = 1;
	namedWindow("gaussian");
	createTrackbar("sigma", "gaussian", &start, 8, on_level_change, (void*)&src);   // 가우시안 시그마 범위(1~8) 지정하영 트랙바 생성
	on_level_change(0, (void*)&src);                                                //프로그램 실행 시 영상이 제대로 출력되도록 강제로 on_level_change 함수 호출

	if (waitKey(0) == 13)                                                    
		imwrite("blurred.bmp", gaussian);

}

void on_level_change(int pos, void* userdata) {                              //트랙바에서 가우시안 시그마 값을 조정하여 화면에 출력
	src = *(Mat*)userdata;                                                   //void* 타입의 인자 userdata를 Mat* 타입으로 형변환 후 src 변수로 참조
	int kernel_size = (8 * pos + 1);                                         //가우시안 필터 마스크 크기는 보통 8*시그마 +1로 결정
	GaussianBlur(src, gaussian, Size(kernel_size, kernel_size), 0);          //가우시안 필터 적용
	cout << "시그마 : " <<pos << endl;                                       //가우시안 시그마 값 화면에 출력
	imshow("gaussian", gaussian);                                            //가우시안 필터가 적용된 이미지 출력
}


void unsharp_mask() {                                                        //언샤프닝 함수
	imread("blurred.bmp");                                                   //가우시안 필터(시그마=4) 이미지 불러오기

	float alpha = 1.f;                                                       //alpha에 1.0을 지정하여 날카로운 성분을 그대로 한 번 더함 >> 날카로운 정도 
	unsharp = (1 + alpha) * src - alpha * gaussian;                          //언샤프닝 공식 대입
	imshow("unsharp", unsharp);                                              //언샤프닝된 이미지 출력
	
	waitKey();
	destroyAllWindows();
}
