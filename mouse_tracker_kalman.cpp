#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

const int winHeight = 600;
const int winWidth = 800;

CvPoint mousePosition = cvPoint(winWidth >> 1, winHeight >> 1);

// mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param) {
  if (event == CV_EVENT_MOUSEMOVE) {
    mousePosition = cvPoint(x, y);
  }
}

int main(void) {
  // 1.kalman filter setup
  const int stateNum = 4;
  const int measureNum = 2;
  CvKalman *kalman =
      cvCreateKalman(stateNum, measureNum, 0); // state(x,y,detaX,detaY)
  CvMat *process_noise = cvCreateMat(stateNum, 1, CV_32FC1);
  CvMat *measurement = cvCreateMat(measureNum, 1, CV_32FC1); // measurement(x,y)
  CvRNG rng = cvRNG(-1);
  float A[stateNum][stateNum] = {// transition matrix
                                 1, 0, 1, 0, 0, 1, 0, 1,
                                 0, 0, 1, 0, 0, 0, 0, 1};

  memcpy(kalman->transition_matrix->data.fl, A, sizeof(A));
  cvSetIdentity(kalman->measurement_matrix,
                cvRealScalar(1)); // H usually set as I
  cvSetIdentity(kalman->process_noise_cov,
                cvRealScalar(1e-5)); // Q usually smaller than R
  cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(1e-1)); // R
  cvSetIdentity(kalman->error_cov_post,
                cvRealScalar(1)); // P(k)=(I-K(k)*H)*P'(k), P(0) can't set as 0,
                                  // o.w. it will have no noise in the
                                  // enviroment
  // initialize post state of kalman filter at random
  cvRandArr(&rng, kalman->state_post, CV_RAND_UNI, cvRealScalar(0),
            cvRealScalar(winHeight > winWidth
                             ? winWidth
                             : winHeight)); // x(k)=x'(k)+K(k)*(z(k)-H*x'(k))

  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 1, 1);

  cvNamedWindow("kalman");
  cvSetMouseCallback("kalman", mouseEvent);
  IplImage *img = cvCreateImage(cvSize(winWidth, winHeight), 8, 3);
  while (1) {
    // 2.kalman prediction
    const CvMat *prediction = cvKalmanPredict(kalman, 0);
    CvPoint predict_pt =
        cvPoint((int)prediction->data.fl[0], (int)prediction->data.fl[1]);

    // 3.update measurement
    measurement->data.fl[0] = (float)mousePosition.x;
    measurement->data.fl[1] = (float)mousePosition.y;

    // 4.update
    cvKalmanCorrect(kalman, measurement);

    // draw
    cvSet(img, cvScalar(255, 255, 255, 0));
    cvCircle(img, predict_pt, 5, CV_RGB(0, 255, 0),
             3); // predicted point with green
    cvCircle(img, mousePosition, 5, CV_RGB(255, 0, 0),
             3); // current position with red
    char buf[256];
    snprintf(buf, 256, "predicted position:(%3d,%3d)", predict_pt.x,
             predict_pt.y);
    cvPutText(img, buf, cvPoint(10, 30), &font, CV_RGB(0, 0, 0));
    snprintf(buf, 256, "current position :(%3d,%3d)", mousePosition.x,
             mousePosition.y);
    cvPutText(img, buf, cvPoint(10, 60), &font, CV_RGB(0, 0, 0));

    cvShowImage("kalman", img);
    char key = cvWaitKey(3);
    if (key == 27) { // esc
      break;
    }
  }
  cvReleaseImage(&img);
  cvReleaseKalman(&kalman);
  return 0;
}
