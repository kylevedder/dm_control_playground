#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

struct Point {
  float px_;
  float py_;
  float pz_;
  float pr_;
  Point(float px, float py, float pz, float pr)
      : px_(px), py_(py), pz_(pz), pr_(pr) {}
};

void VerifyFile(std::string filename) {
  // allocate 4 MB buffer (only ~130*4*4 KB are needed)
  int32_t num = 10000000;
  float *data = (float *)malloc(num * sizeof(float));

  // pointers
  float *px = data + 0;
  float *py = data + 1;
  float *pz = data + 2;
  float *pr = data + 3;

  std::vector<Point> points;

  // load point cloud
  FILE *stream = fopen(filename.c_str(), "rb");
  num = fread(data, sizeof(float), num, stream) / 4;
  for (int32_t i = 0; i < num; i++) {
    points.push_back(Point(*px, *py, *pz, *pr));
    px += 4;
    py += 4;
    pz += 4;
    pr += 4;
  }
  fclose(stream);
  std::cout << "Read " << points.size() << " points" << std::endl;
  // std::cout << "Last point: " << points.back().px_ << ", " << points.back().py_
  //           << ", " << points.back().pz_ << ", " << points.back().pr_
  //           << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <.bin file> [.bin file ...]"
              << std::endl;
    return 1;
  }

  for (int i = 1; i < argc; ++i) {
    VerifyFile(argv[i]);
  }
}