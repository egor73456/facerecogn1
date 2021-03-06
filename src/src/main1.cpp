#include "lib.hpp"

// обработка аргументов командной строки
void check_args(int argc, char const **argv) {
  if (argc != 3 && argc != 1) {
    std::cerr << "Usage: " << argv[0]
              << " [\"/caffe/config/path\" \"/caffe/weights/path\"]"
              << std::endl;
    exit(1);
  }
  if (argc == 3) {
    caffeConfigFile = argv[1];
    caffeWeightFile = argv[2];
  }
}

// загрузить конфигурацию и веса сети из файлов
cv::dnn::Net open_net(char const **argv) {
  try {
    return cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
  } catch (std::exception) {
    std::cerr << "[" << argv[0] << "] Файл(ы) не найден(ы) или поврежден(ы)!"
              << std::endl;

    std::cerr << argv[1] << " " << argv[2] << std::endl;
    exit(1);
  }

}

int main(int argc, char const **argv) {
  check_args(argc, argv);
  int hue = 100, sat = 100, val = 100;
  int inverse_colors = 0;
  cv::Mat drawing_layer, frame, text_n_boxes;
  // cv::createButton("Inverse colors", button_callback, &inverse_colors,
  // cv::QT_PUSH_BUTTON);
  std::atomic<bool> stop_flag = 0;
  cv::dnn::Net net = open_net(argv);
  cv::VideoCapture capture{0};
  if (!capture.isOpened())
    return -1;
  capture >> frame;
  drawing_layer.create(frame.size(), frame.type());
  text_n_boxes.create(frame.size(), frame.type());
  std::thread detect_th{[&net, &frame, &text_n_boxes, &stop_flag] {
    detectFaceOpenCVDNN(net, frame, text_n_boxes, stop_flag);
  }};
  for (;;) {
    cv::Mat tmp;
    {
      std::lock_guard a{main_im_access};
      capture >> frame;
      cv::copyTo(frame, tmp, cv::noArray());
    }
    process_im(tmp, hue, sat, val, inverse_colors, drawing_layer);
    {
      std::lock_guard b{im_access};
      tmp += text_n_boxes;
    }

    auto diff = (curt_time = std::chrono::system_clock::now()) - prev_time;
    if (diff.count() > 0)
      fps =
          1'000 /
          (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count());
    if (fps <= fps_max) {
      cv::imshow("Video", tmp);
      prev_time = curt_time;
    } else
      fps = fps_max;

    auto key = cv::waitKey(1);
    if (key == 27)
      break;
  }
  stop_flag = 1;
  detect_th.join();
  return 0;
}