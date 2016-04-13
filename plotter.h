#ifndef __PLOTTER_H__
#define __PLOTTER_H__

class Plotter {
    public:
    
    void plotData(std::vector<double> data) {
        FILE *pipe = popen("gnuplot -persist" , "w");

        if (pipe != NULL) {

            fprintf(pipe, "set style line 5 lt rgb 'cyan' lw 3 pt 6 \n");
            fprintf(pipe, "plot '-' with linespoints ls 5 \n");

            for (int i=0; i<data.size(); i++) {
                fprintf(pipe, "%lf %lf\n", double(i), data[i]);
            }
            fprintf(pipe, "e");

            fflush(pipe);
            pclose(pipe);
        }
        else {
            std::cout << "Could not open gnuplot pipe" << std::endl;
        }
    }
};

#endif