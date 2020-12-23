//
//  ViewController.mm
//  squeenzecnn
//
//  Created by dang on 2017/7/28.
//  Copyright © 2017年 dang. All rights reserved.
//

#import "ViewController.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#include <algorithm>
#include <functional>
#include <vector>
#include "ncnn/ncnn/net.h"
#include "ncnn/ncnn/platform.h"
#include "ncnn/ncnn/mobilenet_v01.id.h"
//#include "mobilenet_v01.id.h"
//#include "mo"
//#include "mobilenet_v01.id.h"
//#include "mobilenetv2_yolov3.id.h"
//#include "Y"
@interface ViewController ()
@property (strong, nonatomic) IBOutlet UILabel *resultLabel;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

}

NSString *print_topk(const std::vector<float>& cls_scores, int topk, std::vector<std::string> labels)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    NSString *result = @"";
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        //        fprintf(stderr, "%d = %f\n", index, score);
        std::string label = labels[index];
        NSString *string = [NSString stringWithUTF8String:label.c_str()];
        result = [result stringByAppendingFormat:@"%@ %f \n", string, score];
        NSLog(@"label: %@ prob: %f", string, score);
    }
    return result;
}

int load_labels(std::string path, std::vector<std::string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");

    while (!feof(fp))
    {
        char str[1024];
        fgets(str, 1024, fp);
        std::string str_s(str);

        if (str_s.length() > 0)
        {
            for (int i = 0; i < str_s.length(); i++)
            {
                if (str_s[i] == ' ')
                {
                    std::string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    return 0;
}

- (IBAction)predict:(id)sender {
    // load image
    UIImage* image = [UIImage imageNamed:@"test.jpg"];
    // get rgba pixels from image
    int w = image.size.width;
    int h = image.size.height;
    fprintf(stderr, "%d x %d\n", w, h);
    unsigned char* rgba = new unsigned char[w*h*4];
    {
        CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
        CGContextRef contextRef = CGBitmapContextCreate(rgba, w, h, 8, w*4,
                                                        colorSpace,
                                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);

        CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
        CGContextRelease(contextRef);
    }

    
//    NSURL *fileURl = [[NSBundle mainBundle] URLForResource:@"mobilenet_v01" withExtension:@"bin"];
    // init net
    ncnn::Net net;
    {
        
//        int r0= net.load_param_bin("mobilenet_v01.param.bin");
//        NSData *da = [NSData dataWithContentsOfFile:paramPath];
//        Byte *testByte = (Byte *)[da bytes];
//        FILE *fp = fopen("mobilenet_v01.param.bin","rb");
       
//        net.load_param_bin(<#const char *protopath#>)
//        int r0 = net.load_param([paramPath UTF8String]);
        NSString *paramPath = [[NSBundle mainBundle] pathForResource:@"mobilenet_v01" ofType:@"param"];
        NSString *binPath = [[NSBundle mainBundle] pathForResource:@"mobilenet_v01" ofType:@"bin"];
        //NSLog(@"paramPath=%@==bin=%@",paramPath,binPath);
        int r0 = net.load_param([paramPath UTF8String]);
        int r1 = net.load_model([binPath UTF8String]);
//        int r1 = net.load_model([binPath UTF8String]);
       
        
        fprintf(stderr, "net load %d %d \n", r0, r1);
    }

    std::vector<std::string> labels;
    NSString *labelPath = [[NSBundle mainBundle] pathForResource:@"words" ofType:@"txt"];
    load_labels([labelPath UTF8String], labels);
    NSLog(@"labelPath=%@",labelPath);
    // run forward
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(2);

//        ncnn::Mat;
       ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2BGR, w, h, 320, 320);
        const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
        const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ex.input(mobilenet_v01_param_id::BLOB_data, in);
        ex.extract(mobilenet_v01_param_id::BLOB_output, out);
        NSLog(@"w=%d===h==%d==c==%d==result=%zu=====%f",out.w, out.h, out.c, out.cstep, float(out[1]));
    }

    // get prob
//    std::vector<float> cls_scores;
//    {
//        cls_scores.resize(out.c);
//        for (int j=0; j<out.c; j++)
//        {
//            const float* prob = out.data + out.cstep * j;
//            cls_scores[j] = prob[0];
//        }
//    }
//
//    NSString *result = print_topk(cls_scores, 3, labels);
//    self.resultLabel.text = result;
//    // clean
//    delete[] rgba;
}

@end
