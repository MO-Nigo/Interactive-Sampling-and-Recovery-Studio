from PyQt5.QtWidgets import QFileDialog

import pandas as pd
import time
import numpy as np
from pyqtgraph.Qt import QtCore
import scipy.signal as sig
from scipy.interpolate import interp1d



import random


#Global variable
f_max = 1

class AppLogic():
        def __init__(self, ui_instance): 
            self.ui_instance = ui_instance
            self.ui_instance.tabWidget.setCurrentIndex(0)
            self.sampled_data = None
            self.sampled_points = None
            self.t = None
            self.ploted_signal = []
            self.signals = {}
            self.max_frequancy_composer = {}
            self.signal_names = []
            self.f_max = 0
            self.previous_slider_value = 0
            self.inter_amplitude_with_noise = []



        def load_signal(self):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self.ui_instance, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
            print(file_name)
            if file_name:
                try:
                    data = pd.read_csv(file_name)
                    time_data = data.iloc[:, 0].values
                    amplitude_data = data.iloc[:, 1].values
                    time_data = time_data - time_data[0] # Normalize time_data to start at 0
                    self.ui_instance.plotSignal.clear()
                    self.ui_instance.plotSample.clear()
                    plot_data = self.ui_instance.plotSignal.plot(time_data, amplitude_data, pen='b')
                    plot_data.setData(time_data[:1001], amplitude_data[:1001])

                    self.f_max=10 #change according the file which we know it's f_sample
                    
                    self.ui_instance.update_labelRMax(self.f_max)                   
                    self.sampled_data = (time_data, amplitude_data)
                    self.ui_instance.sliderHz.setMinimum(1)
                    self.ui_instance.sliderHz.setMaximum(int(4*self.f_max))
                except Exception as e:
                    print(f"Error loading the CSV file: {str(e)}")
            else:
                print("No file selected")


        def sample_and_plot(self):
            if self.sampled_data is not None:
                if self.sampled_points is not None:
                    self.ui_instance.plotSignal.removeItem(self.sampled_points)
                time_data, amplitude_data = self.sampled_data
                f_sample = self.ui_instance.sliderHz.value()
                self.ui_instance.update_labelSlider(f_sample)

                sampled_time = np.arange(
                    0,7.984, 1 / f_sample)  # the upper limit is included
                sampled_amplitude = np.interp(sampled_time, time_data,amplitude_data)
                interpolation_factor = 1001
                interpolated_time = np.linspace(sampled_time.min(), sampled_time.max(), interpolation_factor)
                interpolated_amplitude = []
                for t in interpolated_time:
                    sinc_values = np.sinc((sampled_time - t) * f_sample)
                    interpolated_value = np.sum(sampled_amplitude * sinc_values) / np.sum(sinc_values)
                    interpolated_amplitude.append(interpolated_value)



                # Save the interpolated data to a CSV file
                error = [abs(n) - abs(m) for n, m in zip(amplitude_data, interpolated_amplitude)]

                # Clear and plot the interpolated data
                self.ui_instance.plotSample.clear()
                # self.ui_instance.plotSignal.plot(time_data[:len(noisy_amplitude_data)], noisy_amplitude_data, pen='g')
                plot_data = self.ui_instance.plotSample.plot(time_data, interpolated_amplitude, pen='r')
                self.sampled_points = self.ui_instance.plotSignal.scatterPlot(sampled_time, sampled_amplitude,
                                                                              pen=None,
                                                                              symbol='o', symbolPen='r',
                                                                              symbolBrush='r',  symbolSize=5)



                self.ui_instance.plotSignal.replot()
                self.ui_instance.plotSample.replot()

                # Plot the error on the plotError widget
                self.ui_instance.plotError.clear()

                error_plot = self.ui_instance.plotError.plot(time_data[:900], error[:900], pen='r')
                self.error_plot_data = error_plot


        #Remove Signal
        def remove_signal(self):
            self.ui_instance.plotSignal.clear()
            self.ui_instance.plotSample.clear()
            self.ui_instance.plotError.clear()
            self.ui_instance.labelRMax.clear()
            self.ui_instance.sliderHz.setValue(0)
            self.ui_instance.sliderNoise.setValue(0)




        def create_and_plot_signal(self):
            self.ui_instance.plotBefore.clear()
            # Retrieve values from user input
            name = self.ui_instance.editName.text()
            freq_text = self.ui_instance.editFreq.text()
            amp_text = self.ui_instance.editAmp.text()
            phase_text = self.ui_instance.editPhase.text()
            # Check if the input fields are not empty
            if freq_text and amp_text and phase_text:
                try:
                    # Attempt to convert the input to floats
                    frequency = float(freq_text)
                    amplitude = float(amp_text)
                    phase_shift = float(phase_text)

                    # Generate the time values and signal
                    self.t = np.linspace(0, 7.984, 1001)
                    self.t = np.around(self.t, 3)
                    signal = amplitude * np.sin(2 * np.pi * frequency * self.t + phase_shift)
                    signal = np.around(signal, 5)
                    self.ploted_signal.append(signal)
                    self.signals[name] = signal
                    self.max_frequancy_composer[name] = frequency
                    self.signal_names.append(name)

                    # Plot the signal
                    self.ui_instance.plotBefore.plot(self.t, signal, pen='b')
                except ValueError:
                    pass
            else:
                pass



        def plot_noise_signal(self):
            if self.sampled_data is not None:
                if self.sampled_points is not None:
                    self.ui_instance.plotSignal.removeItem(self.sampled_points)
                time_data, amplitude_data = self.sampled_data
                f_sample = self.ui_instance.sliderHz.value()
                self.ui_instance.update_labelSlider(f_sample)

                # Calculate the desired oversampling factor
                oversampling_factor = 1

                # Calculate the desired sampling period Ts (inverse of f_sample)
                Ts = 1.0 / f_sample

                T = (time_data[-1] - time_data[0]) / (len(time_data) - 1)

                # Upsample the amplitude data to increase its length by oversampling_factor
                inter_amplitude = []
                for i in range(len(amplitude_data)):
                    for j in range(oversampling_factor):
                        t = time_data[i] + j * T / oversampling_factor
                        sinc_values = np.sinc((time_data - t) / T)
                        interpolated_value = np.sum(amplitude_data * sinc_values) / np.sum(sinc_values)
                        inter_amplitude.append(interpolated_value)

                self.inter_amplitude_with_noise = inter_amplitude
                # Check if the slider value has changed
                if  self.up_data():
                        current_slider_value = self.ui_instance.sliderNoise.value()


                            # Add noise because the slider value increased
                        noise_percentage = current_slider_value
                        noise_std = (noise_percentage / 100.0)
                        noise_std = max(0, noise_std)
                        if current_slider_value <= 3:
                            self.inter_amplitude_with_noise=inter_amplitude
                        else:

                            self.inter_amplitude_with_noise = inter_amplitude + np.random.normal(0, noise_std,
                                                                                                      len(inter_amplitude))
                            # Update the previous slider value to the current value
                        self.previous_slider_value = current_slider_value

                # Calculate the corresponding time values for the interpolated amplitude data
                inter_time = np.linspace(time_data[0], time_data[-1], len(inter_amplitude))

                # Calculate the number of sample points based on Ts
                num_sample_points = int((inter_time[-1] - inter_time[0]) / Ts)

                # Use np.arange to create sampled_time with num_sample_points
                sampled_time = np.arange(start=inter_time[0], stop=inter_time[0] + num_sample_points * Ts, step=Ts,
                                         dtype=float)

                # Ensure that sample_indices are within the bounds of the inter_amplitude array
                sample_indices = np.searchsorted(inter_time, sampled_time)
                sample_indices = np.clip(sample_indices, 0, len(inter_amplitude) - 1)

                # Extract the corresponding values from inter_amplitude
                sampled_amplitude = np.array(self.inter_amplitude_with_noise)[sample_indices]

                noise_percentage = self.ui_instance.sliderNoise.value()
                self.ui_instance.labelNoise2.setText(f"{noise_percentage} %")







                # Calculate the sinc interpolation
                interpolation_factor = len(inter_amplitude)
                interpolated_time = np.linspace(sampled_time.min(), sampled_time.max(), interpolation_factor)
                interpolated_amplitude = []
                for t in interpolated_time:
                    sinc_values = np.sinc((sampled_time - t) * f_sample)
                    interpolated_value = np.sum(sampled_amplitude * sinc_values) / np.sum(sinc_values)
                    interpolated_amplitude.append(interpolated_value)
                error = [n - m for n, m in zip(inter_amplitude[:len(interpolated_amplitude)], interpolated_amplitude)]



                # Clear and plot the interpolated data
                self.ui_instance.plotSample.clear()
                self.ui_instance.plotSignal.plot(inter_time, self.inter_amplitude_with_noise, pen='b')
                plot_data = self.ui_instance.plotSample.plot(interpolated_time, interpolated_amplitude, pen='r')
                self.sampled_points = self.ui_instance.plotSignal.scatterPlot(sampled_time, sampled_amplitude,
                                                                              pen=None,
                                                                              symbol='o', symbolPen='r',
                                                                              symbolBrush='r', symbolSize=5)


                self.ui_instance.plotSample.replot()

                # Plot the error on the plotError widget
                self.ui_instance.plotError.clear()
                error_plot = self.ui_instance.plotError.plot(time_data, error, pen='g')
                self.error_plot_data = error_plot

                # # Modify the error to set small errors to zero when slider is at or above 2*f_max
                # if f_sample >= 2 * self.f_max:
                #     small_error_threshold = 1e-4  # Adjust this threshold as needed
                #     error[abs(error) < small_error_threshold] = 0


        def up_data(self):
           if self.ui_instance.sliderNoise.value() != self.previous_slider_value:
               flag=1
           else:
               flag=0

           return flag


        def composer(self):
            self.ui_instance.plotAfter.clear()
            self.ui_instance.plotBefore.clear()

            if self.signals:
                # Sum all the signals in the list
                self.mix = np.sum(self.ploted_signal, axis=0)
                self.ui_instance.plotAfter.plot(self.t, self.mix, pen='r')
                # print(self.signal_names)
                self.ui_instance.signal_composer.clear()
                self.ui_instance.signal_composer.addItems(self.signal_names)
                max_feq =str(max(self.max_frequancy_composer.values()))
                self.ui_instance.composer_freq(max_feq)

                self.ui_instance.editName.clear()
                self.ui_instance.editFreq.clear()
                self.ui_instance.editAmp.clear()
                self.ui_instance.editPhase.clear()

        def remove_signal_tab2(self):
            selected_signal_name = self.ui_instance.signal_composer.currentText()

            if selected_signal_name != "":
                # self.ui_instance.signal_composer.removeItem(selected_signal_name)
                selected_signal_value = np.array(self.signals[selected_signal_name])
                self.mix = np.array(self.mix)
                self.mix = self.mix - selected_signal_value
                self.mix = self.mix.tolist()
                self.ui_instance.signal_composer.removeItem(self.ui_instance.signal_composer.currentIndex())
                del self.signals[selected_signal_name]
                del self.max_frequancy_composer[selected_signal_name]
                self.signal_names.remove(selected_signal_name)
                self.ui_instance.plotAfter.clear()

                if len(self.max_frequancy_composer) != 0:
                    self.f_max = str(max(self.max_frequancy_composer.values()))
                else:
                    self.f_max = "0"

                self.ui_instance.composer_freq(self.f_max)

                if self.signal_names:
                    self.ui_instance.plotAfter.plot(self.t, self.ploted_signal, pen='b')
            else:
                self.ui_instance.plotAfter.clear()
                self.ui_instance.labelRRFreq.setText("   ")

        def plot_mix(self):
            self.ui_instance.plotSignal.clear()  # Clear the PlotWidget
            self.ui_instance.tabWidget.setCurrentIndex(0)
            self.ui_instance.plotAfter.clear()
            self.ui_instance.labelRRFreq.setText("   ")

            self.f_max = max(self.max_frequancy_composer.values())
            self.t=np.array(self.t)
            df = pd.DataFrame({'Time': self.t, 'Amplitude': self.mix})
            self.ui_instance.plotSignal.plot(self.t, self.mix, pen='b')
            self.ui_instance.update_labelRMax(self.f_max)
            self.ui_instance.signal_composer.clear
            self.signal_names.clear()

            self.max_frequancy_composer.clear()



            self.ui_instance.composer_freq(" ")
            self.ploted_signal.clear()

            self.ui_instance.sliderHz.setMinimum(1)
            self.ui_instance.sliderHz.setMaximum(int(4 *self.f_max))
            self.sampled_data = (self.t, df['Amplitude'])
            df.to_csv("signal_composer" + '.csv', index=False)

