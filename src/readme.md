# FACERECOGN
```c++
src/lib:
  // коэффициент преобразования размеров изображений(параметр сети распознования)
  extern const double inScaleFactor;
  // среднее значение(параметр сети распознования)
  extern const cv::Scalar meanVal;
  // нижняя граница точности совпадения результата нейронной сети(все, что ниже
  // будет игнорироваться)
  extern const float confidenceThreshold;

  // mutex для доступа к матрице изображения text_n_boxes(название переменной из
  // main1.cpp) или ret(название из detectOpenCVDNN)
  extern std::mutex im_access;
  // mutex для доступа к  матрице изображения frameOpenCVDNN(название из
  // detectOpenCVDNN)
  extern std::mutex main_im_access;

  // путь до файла конфигурации нейронки
  extern std::string caffeConfigFile;
  // путь до файла весов нейронки
  extern std::string caffeWeightFile;

  // предыдущее и текущее время замера(для рассчета FPS)
  extern std::chrono::_V2::system_clock::time_point prev_time, curt_time;
  // среднее значение FPS за промежуток обновления кадра и максимальное значение
  // FPS(для контроля ограничения FPS)
  extern int fps, fps_max;
  // цвет, который будет использован для рисования при помощи ЛКМ
  extern cv::Scalar draw_color;
  // структура для проброса аргументов через безымянный указатель в функцию
  // mouseHandler
  struct BreakThroughContainer {
    // ссылки на матрицу рисования и матрицу основного изображения
    cv::Mat &draw, &frame;
    // указатель на объект, описывающий предудущую позицию мыши
    std::unique_ptr<cv::Point> prev;
    // состояние рисовалки (-1 - стирать, 0 - бездействие, 1 - рисовать)
    int drawing;
  };
  // функция обработчик изображения, которая отвечает за распознавание лица в
  // матрице @frameOpenCVDNN, и рисование прямоугольника на матрице @ret. Следует
  // вызывать в отдельномм потоке, потому что функция перехватывает управление
  extern void detectFaceOpenCVDNN(cv::dnn::Net &net, cv::Mat &frameOpenCVDNN,
                                  cv::Mat &ret, std::atomic<bool> &stop);
  // функция обработки изображения по изменению насыщенности(@sat), тона(@hue) и
  // яркости(@val) цвета. так же обрабатвает изображение в случае необходимости
  // инверсии цветов и накладывает слой рисования на основное изображение
  extern void process_im(cv::Mat &frame, int &hue, int &sat, int &val,
                        int &inverse_colors, cv::Mat &drawing_layer);
  // фкнкция обработчик событий мыши
  extern void mouseHandler(int event, int x, int y,
                          [[maybe_unused]] int flags, void *param);
src/main1:
  // обработка аргументов командной строки
  void check_args(int argc, char const **argv);
  // загрузить конфигурацию и веса сети из файлов
  cv::dnn::Net open_net(char const **argv);
  // точка входа
  int main(int argc, char const **argv);
  ```c++