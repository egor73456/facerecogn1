#include "lib.hpp"

const double inScaleFactor = 1.0;
const cv::Scalar meanVal{104.0, 177.0, 123.0};
const float confidenceThreshold = 0.4f;

std::mutex im_access, main_im_access;

std::string caffeConfigFile =
    "/data/Source/opencvfacerecogn-project/data/deploy.prototxt";
std::string caffeWeightFile = "/data/Source/opencvfacerecogn-project/data/"
                              "res10_300x300_ssd_iter_140000_fp16.caffemodel";

std::chrono CHRONOV2 system_clock::time_point prev_time, curt_time;
int fps = 0, fps_max = 40;
cv::Scalar draw_color{0, 0, 255};

void detectFaceOpenCVDNN(cv::dnn::Net &net, cv::Mat &frameOpenCVDNN,
                         cv::Mat &ret, std::atomic<bool> &stop) {
  cv::Size size;
  while (stop)
    ;
  {
    std::lock_guard a{main_im_access};
    size = frameOpenCVDNN.size();
  }
  for (;;) {
    cv::Mat inputBlob;
    {
      std::lock_guard a{main_im_access};
      inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size,
                                         meanVal, false, false);
    }
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F,
                         detection.ptr<float>());
    {
      std::lock_guard a{im_access};
      ret = cv::Mat::zeros(ret.size(), ret.type());
    }
    for (int i = 0; i < detectionMat.rows; i++) {
      float confidence = detectionMat.at<float>(i, 2);
      if (confidence > confidenceThreshold) {

        int x1 = static_cast<int>(detectionMat.at<float>(i, 3) *
                                  static_cast<float>(size.width));
        int y1 = static_cast<int>(detectionMat.at<float>(i, 4) *
                                  static_cast<float>(size.height));
        int x2 = static_cast<int>(detectionMat.at<float>(i, 5) *
                                  static_cast<float>(size.width));
        int y2 = static_cast<int>(detectionMat.at<float>(i, 6) *
                                  static_cast<float>(size.height));
        {
          std::lock_guard a{im_access};
          cv::rectangle(ret, cv::Point(x1, y1), cv::Point(x2, y2),
                        cv::Scalar(0, 255, 0), 2, 4);
          std::stringstream ss;
          ss.str("");
          ss << confidence;
          cv::putText(ret, ss.str(), cv::Point(x1, y1), cv::FONT_HERSHEY_PLAIN,
                      1.0, CV_RGB(0, 255, 0), 2.0);
        }
      }
    }
    if (stop)
      break;
  }
}

void process_im(cv::Mat &frame, int &hue, int &sat, int &val,
                int &inverse_colors, cv::Mat &drawing_layer) {
  cv::Mat hsv_buff;
  cv::cvtColor(frame, hsv_buff, cv::COLOR_BGR2HSV);
  auto vals = hsv_buff.mul(cv::Scalar(hue * 0.01, sat * 0.01, val * 0.01));
  cv::cvtColor(vals, frame, cv::COLOR_HSV2BGR);
  std::stringstream ss;
  ss.str("");
  ss << "FPS: " << fps;
  if (inverse_colors)
    frame = -frame + cv::Scalar(255, 255, 255);
  cv::putText(frame, ss.str(), {0, 30}, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0,
              CV_RGB(255, 255, 255), 2.0);
  frame += drawing_layer;
}

// void button_callback(int, void *inv) { *reinterpret_cast<bool*>(inv) =
// !*reinterpret_cast<bool*>(inv); }

void mouseHandler(int event, int x, int y, [[maybe_unused]] int flags,
                  void *param) {
  BreakThroughContainer &args =
      *reinterpret_cast<BreakThroughContainer *>(param);
  cv::Mat &draw = args.draw;
  cv::Point &prev =
      args.prev ? *args.prev : *(args.prev = std::make_unique<cv::Point>(x, y));
  if (event == cv::EVENT_LBUTTONDOWN)
    args.drawing = 1;
  if (event == cv::EVENT_LBUTTONUP)
    args.drawing = 0;
  if (event == cv::EVENT_RBUTTONDOWN)
    args.drawing = -1;
  if (event == cv::EVENT_RBUTTONUP)
    args.drawing = 0;
  if (event == cv::EVENT_MBUTTONUP) {
    std::lock_guard a{main_im_access};
    auto tmp = args.frame.at<cv::Vec3b>(y, x);
    draw_color = {static_cast<double>(tmp[0]), static_cast<double>(tmp[1]),
                  static_cast<double>(tmp[2])};
  }
  if (args.drawing == 1)
    cv::line(draw, {x, y}, prev, draw_color, 3);
  if (args.drawing == -1)
    cv::ellipse(draw, {x, y}, {40, 40}, 0, 0, 360, cv::Scalar(0, 0, 0), -1);
  prev = {x, y};
};