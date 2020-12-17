#include <jni.h>
#include "secedct.h"

int seceTime = 0;

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_imageenhancementandroid_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_imageenhancementandroid_MainActivity_seceFromJNI(
        JNIEnv* env,
        jobject instance,
        jintArray rawImg, jint w, jint h) {

    jint *cbuf = env->GetIntArrayElements(rawImg, JNI_FALSE );
    if (cbuf == NULL) {
        return 0;
    }

    Mat imgData(h, w, CV_8UC4, (unsigned char *) cbuf);
    /*图像处理开始*/
    cvtColor(imgData,imgData,CV_BGRA2BGR);
    seceTime = calTime(contrastEnhancement, imgData, imgData);
    cout << "seceTime: " << seceTime << endl;
    cvtColor(imgData,imgData,CV_BGR2BGRA);

    /*图像处理结束*/
    uchar *ptr = imgData.data;
    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, (const jint *)ptr);
    env->ReleaseIntArrayElements(rawImg, cbuf, 0);

    return result;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_imageenhancementandroid_MainActivity_timeFromJNI(JNIEnv *env, jobject thiz) {
    // TODO: implement timeFromJNI()
    std::string time = to_string(seceTime);
    return env->NewStringUTF(time.c_str());
}









