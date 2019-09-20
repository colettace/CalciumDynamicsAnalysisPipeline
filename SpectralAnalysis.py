#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


from pathlib import Path


# In[3]:


data_dir = Path( '../pipeline_output').resolve()
input_videos_dir = data_dir / "bg_subtr_videos"
#input_timestamps_dir = data_dir / "timestamp_datadump"
#fft_histogram_output_dir = data_dir / "fft_histograms"
#pca_output_dir = data_dir / "PCA_plots"
#avg_power_plot_dir = data_dir / "avg_power_plots"
#output_dir = data_dir / "isochronic_color_maps"
output_dir = data_dir / "fft_data"


# In[4]:


output_subdir_names = [ "fft_data" ]#, "fft_histograms", "average_power_plots" ]

for subdir_name in output_subdir_names:
    subdir = data_dir / subdir_name
    if not subdir.exists():
        print( f"Making directory {subdir.resolve()}")
        subdir.mkdir()


# In[5]:


# Rebuild file list to figure out what's next
input_data = []
#print( "****************************")
input_video_paths = list(input_videos_dir.glob("*_BG_SUBTR.tif"))
for input_video_path in input_video_paths:

    stem = input_video_path.name[:-13]
    #print( stem )
    
    # Already done?
    #output_path = output_dir / f"{stem}_isochron_color_maps"
    #if output_path.exists():
    #    print( f"Skipping \"{str(output_path)}, since it already exists.")
    #    continue
    
    ampl_path = output_dir / f"{stem}_FFT_AMPLITUDE.npy"
    phase_path = output_dir / f"{stem}_FFT_PHASE.npy"
    
    #timestamp_path = input_timestamps_dir / f"{stem}_timestamp_data.csv"
    #if not timestamp_path.exists():
    #    print( f"WARNING: {str(timestamp_path)} doesn't exist, please regenerate.")
    #    continue
        
    #histogram_path = fft_histogram_output_dir / f"{stem}_fft_histogram.tif"
    #avg_power_plot_path = avg_power_plot_dir / f"{stem}_avg_power_plot.tif"
    
    paths = [ ampl_path, phase_path ]#, histogram_path ]
    
    if all( [ _.exists() for _ in paths ] ):
        print( f"Skipping \"{str(stem)}\", since amplitude and phase are already saved to disk.")
        continue
    paths.insert( 0, input_video_path )
    
    #pca_path = pca_output_dir / f"{stem}_pca_pixel_similarity.tif"
    #output_path = output_dir / f"{stem}_isochron_colormap"
    
    input_data.append( paths )

print( f"Found {len(input_video_paths)} vidoes, processing {len( input_data ) } of them." )


# In[6]:


def ParallelFFT( inputim, n_workers=20, inverse=False ):
    import numpy as np

    from multiprocessing import Pool
    from numpy.fft import fft, ifft

    transform = ifft if inverse else fft

    nframes, nrows, ncols = inputim.shape
    tmp = np.moveaxis( inputim, 0, 2 )
    tmp = tmp.reshape( -1, nframes )

    p = Pool( n_workers )
    tmp = np.array( p.map( transform, tmp ) )
    p.close()
    p.join()

    tmp = tmp.reshape( nrows, ncols, -1 )
    tmp = np.moveaxis( tmp, 2, 0 )
    if inverse:
        tmp=tmp.real
    return tmp


# In[7]:


def FFTProcessing( input_video_path, ampl_path, phase_path, debug=True ):

    from pathlib import Path
    import time
    if ampl_path.exists() and phase_path.exists():
        print( f'Skipping FFT generation for {str( input_video_path)} since amplitude and phase files already exists.')
        return

    if debug:
        print( "****************************")
        print( "Input video path:", str( input_video_path ) )
        print( "Amplitude data path:", str( ampl_path ) )
        print( "Amplitude data path:", str( phase_path ) )
        t00 = time.time()

    # ========================================================
    # Part 1: Load video from disk 
    from skimage.io import imread, imsave
    import numpy as np

    if debug:
        t0 = time.time()

    orig_vid = imread( str(input_video_path ) )
    
    if debug:
        t1 = time.time()
        print( f"Loaded source video \"{str(input_video_path)}\", shape={orig_vid.shape}, {orig_vid.nbytes/1073741824:0.3f} GB total. Took {t1-t0:0.2f} seconds" )
        
    # ========================================================
    # Part 2: Do FFT
    if debug:
        t0 = time.time()

    fft_im = ParallelFFT( orig_vid )
    del orig_vid
    
    if debug:
        t1 = time.time()
        print( f"FFT transform video into complex128 type ({fft_im.nbytes/1073741824:0.3f} GB total) took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 3: Take amplitude and down convert to 32bit (16??) from 64 to save memory
    if debug:
        t0 = time.time()

    ampl = np.abs(fft_im).astype( np.float32 )
    
    if debug:
        t1 = time.time()
        print( f"Amplitude operation took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 4: Write amplitude to disk
    if debug:
        t0 = time.time()

    np.save( str( ampl_path ), ampl )
    del ampl
    
    if debug:
        t1 = time.time()
        print( f"Saving { str(ampl_path) } to disk took {t1-t0:0.3f} seconds" )

    # ========================================================
    # Part 5: Take phase
    if debug:
        t0 = time.time()

    phase = np.angle( fft_im ).astype( np.float32 )
    del fft_im
    
    if debug:
        t1 = time.time()
        print( f"Phase operation took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 6: Write phase to disk
    if debug:
        t0 = time.time()

    np.save( str(phase_path), phase )
    del phase
    
    if debug:
        t1 = time.time()
        print( f"Saving {str(phase_path)} to disk took {t1-t0:0.3f} seconds" )

    print( f"TOTAL PROCESSING TIME for \"{input_video_path}\" took {time.time()-t00:0.2f} seconds.")
    return


# In[8]:


for paths in input_data:
    FFTProcessing( *paths )


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


from pathlib import Path
# Declare numpy at global scope because I want to use type hints for np.ndarray
import numpy as np 


# In[3]:


data_dir = Path( '../pipeline_output').resolve()
input_fft_dir = data_dir / "fft_data"
input_timestamps_dir = data_dir / "timestamp_datadump"
output_fft_histogram_dir = data_dir / "fft_histograms"
output_pca_dir = data_dir / "pca_plots"
output_avg_power_img_dir = data_dir / "avg_power_plots"
output_isochron_top_level_dir = data_dir / "isochronic_color_maps"


# In[4]:


output_subdirs = [ output_fft_histogram_dir, output_pca_dir,
                  output_avg_power_img_dir, output_isochron_top_level_dir ]

for subdir in output_subdirs:
    if not subdir.exists():
        print( f"Making directory {str(subdir)}")
        subdir.mkdir()


# In[5]:


suffix = "_FFT_AMPLITUDE.npy"


# In[6]:


# Rebuild file list to figure out what's next
input_data = []
#print( "****************************")
input_paths = list(input_fft_dir.glob("*"+suffix))
for input_path in input_paths:

    stem = input_path.name[:-len(suffix)]
    print( stem )

    # Already done?
    #output_isochron_subdir = output_isochron_top_level_dir / f"{stem}_isochron_color_maps"
    #if output_isochron_subdir.exists():
    #    print( f"Skipping \"{stem}, since \"{str(output_isochron_subdir)}\"already exists.")
    #    continue
    
    relevant_paths = [
        input_fft_dir / f"{stem}_FFT_AMPLITUDE.npy",
        input_fft_dir / f"{stem}_FFT_PHASE.npy",
        input_timestamps_dir / f"{stem}_TIMESTAMP_DATA.csv",

        output_fft_histogram_dir / f"{stem}_FFT_HISTOGRAM.pdf",
        output_pca_dir / f"{stem}_PCA_PIXEL_SIMILARITY.pdf",
        output_avg_power_img_dir / f"{stem}_AVG_POWER.tif",
        output_isochron_top_level_dir / f"{stem}_ISOCHRON"
    ]

    have_files = [ _.exists() for _ in relevant_paths ]
    for p, h in zip( relevant_paths, have_files ):
        print( p, h )
    
    relevant_paths.insert( 0, stem )
    input_data.append( relevant_paths )

print( f"Found {len(input_paths)} vidoes, processing {len( input_data ) } of them." )


# In[7]:


def CenterAndClipPhaseAngles( amplitudes : np.ndarray, phases : np.ndarray,
                             lbound_phase_clip_pctile=5, ubound_phase_clip_pctile=95 ) -> np.ndarray:
    """For the given input Fourier amplitudes and phases:
    Figure out the most common phase angle (range= -pi to +pi).
    Perform a phase shift such that the most common phase angle occurs at 0 degrees.
    Clip phases that are way out in the tails of the phase angle bell curve.
    
    amplitude_stack - NON-complex float32 or other with dim={n_frames, n_rows, n_cols}
    phase_stack - NON-complex float32 or other with dim={n_frames, n_rows, n_cols}
    fft_freq_index - the FFT bin index to extract the relevent frequency info
        i.e., amplitude_stack[ fft_freq_index ] and phase_stack[ fft_freq_index ]
        
    return - phase angle array (non-complex data type) with phase shift.
    """
    import numpy as np
    hist_bin_counts, hist_bin_edges = np.histogram( phases, bins=30 )
    
    # Put the negative sign on the angle since we want to shift all
    # angles BACK towards the origin (12 noon) by this amount:
    most_common_angle = hist_bin_edges[ hist_bin_counts.argmax()+1 ] * -1

    from math import cos, sin
    phase_shift = cos( most_common_angle ) + sin( most_common_angle ) * 1j
    # Reconstitute the raw Fourier coefficients
    complex_plane = amplitudes * ( np.cos(phases) + np.sin(phases) * 1j )
    complex_plane *= phase_shift
    new_phases = np.angle( complex_plane )
    wanted_pctiles = [lbound_phase_clip_pctile, ubound_phase_clip_pctile]
    lbound, ubound = np.percentile( new_phases.flat, wanted_pctiles )
    new_phases = np.clip( new_phases, lbound, ubound )
    return new_phases


# In[8]:


def GenerateIsochronFigure(
        amplitudes : np.ndarray,
        phases : np.ndarray,
        rank : int,
        frequency : float,
        fft_bin : int,
        output_figure_path_prefix : Path ) -> None:
    """WARNING!! INPUT ARRAYS ARE MODIFIED!!!
    
    The color of the pixel is a function of the phase angle
    The brightness of the pixel is the square root of its amplitude.
    
    period - the time in miliseconds to complete one cycle for the relevant frequency.
    """

    shifted_phases = CenterAndClipPhaseAngles( amplitudes, phases )
    # Make a copy, because we need unmodified phases for legend
    hue_plane = shifted_phases.copy()
    # Hue = color: 0 = red, 1/6=yellow, 2/6=green, 3/6=cyan
    # 4/6=blue, 5/6=magenta, 6/6 back to red. 
    hue_plane -= hue_plane.min()
    hue_plane /= hue_plane.max()
    # Reverse the colors, per Rostislav request
    hue_plane = 1 - hue_plane
    
    # Saturation => spectrum between 0 (gray) and 1 (color)
    # We want full color unadulterated by gray:
    saturation_plane = np.ones_like( hue_plane )
    
    # Value = how bright is the pixel.
    # 0 => black, 1 => full color
    value_plane = amplitudes
    value_plane -= value_plane.min()
    value_plane /= value_plane.max()
    # Square root accentuates pixels that are brightness > 0.5 and
    # penalizes pixels with brightness < 0.5
    value_plane = np.sqrt( value_plane )
    
    HSV_pixels = np.dstack( [hue_plane, saturation_plane, value_plane] )

    from skimage.color import hsv2rgb
    RGB_pixels = hsv2rgb( HSV_pixels )
    
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots( figsize=(10,8), nrows=2, dpi=300, tight_layout=True )
    period = 1000 / frequency # Convert to miliseconds
    unit_period = period / (2*np.pi)
    title = f"{str(output_figure_path_prefix.name) }\nFFT bin {fft_bin} (period={period:0.1f}ms, freq={frequency:0.2f}Hz)"
    #fig.suptitle( title )

    # Use a color mapping that is circular, i.e.,
    # pi, 3pi, 5pi are all the same color
    colormap = plt.get_cmap('hsv')
    # We have to do this weird thing which is to plot just the phases (1 channel) on
    # the figure first solely for the purpose of getting the colorbar/legend.
    # We then overwrite the RGB pixels (3-channel) into the figure and keep the old colorbar.
    retval = ax1.imshow( shifted_phases, cmap=colormap )
    cbar = fig.colorbar( retval, orientation="horizontal" )

    def format_func( value, ticknumber ):
        #print (deltat, value, deltat/value)
        if value == 0:
            return "Peak Mag"
        return f"{int( round( unit_period * value ))}"
    cbar.ax.xaxis.set_major_formatter( plt.FuncFormatter(format_func) )
    cbar.ax.xaxis.set_label_text( "Time (ms)" )
    ax1.imshow( RGB_pixels )
    ax1.set_axis_off()

    ax0.hist( shifted_phases.flat, bins=30 )
    ax0.set_xlabel( f"Post-clipped distribution of phase angles (radians)" )
    ax0.set_title( title )
    fig.savefig( str( output_figure_path_prefix ) + ".pdf" )
    plt.close( fig )  # Explicitly close to free memory and avoid warning

    from skimage.io import imsave
    RGB_pixels = (RGB_pixels * 255).round().astype( np.uint8 )
    imsave( str( output_figure_path_prefix ) + ".tif", RGB_pixels )
    return


# In[9]:


def ReadTimestamps( input_timestampdata_path : Path ):
    import pandas as pd
    timestamps = pd.read_csv( str(input_timestampdata_path), index_col=0 )
    video_t_i = float( timestamps.head(1)['adj_time'] )
    video_t_f = float( timestamps.tail(1)['adj_time'] ) 

    deltat = video_t_f-video_t_i
    framerate = (video_t_f-video_t_i)*(1000/len(timestamps))
    print( f'TIMESTAMPS: n_timestamps={len(timestamps)}, t1={video_t_i:0.2f}, t2={video_t_f:0.2f}, framerate={framerate:0.2f} ms/frame' )
    return deltat, framerate, timestamps


# In[10]:


def CalculateMeanAmplitudes( ampl ) -> np.ndarray:
    return 10 * np.log10( ampl[1: int(len(ampl)/2) ].mean( axis=(1,2) ) )


# In[11]:


def FFT_PostProcessing( 
        input_vid_fname_stem : str,
        input_fft_ampl_path : Path,
        input_fft_phase_path : Path,
        input_timestampdata_path : Path,
        output_fft_histogram_path : Path,
        output_pca_path : Path,
        output_avg_power_img_path : Path,
        output_isochron_subdir : Path,
        debug=True ):

    import time
    t00 = time.time()

    if debug:
        print( "****************************" )
        print( f"Video \"{input_vid_fname_stem}\"")
        print( "Input amplitude path:", str( input_fft_ampl_path ) )
        print( "Input phase path:", str( input_fft_phase_path ) )
        print( "Input timestamp data path:", str( input_timestampdata_path ) )
        print( "Output FFT histogram path:", str( output_fft_histogram_path ) )
        print( "Output PCA pixel similarity figure path:", str( output_pca_path ) )
        print( "Output FFT average power image path:", str( output_avg_power_img_path ) )
        print( "Output colorized isocron directory", str( output_isochron_subdir ) )
        print( "beginning processing now...\n" )        

    # ========================================================
    # Part 1: Load amplitude and dump average power plot
    import numpy as np
    from skimage.io import imread, imsave
    import matplotlib.pyplot as plt

    ampl = None
    deltat = None
    timestamps = None
    framerate = None
    mean_amplitudes = None
    
    if output_avg_power_img_path.exists():
        print( f'Skipping creation of "{str(output_avg_power_img_path)}" since it already exists.')
    else:
        if debug:
            t0 = time.time()
        ampl = np.load( str( input_fft_ampl_path ) )
    
        if debug:
            t1 = time.time()
            print( f"Loaded amplitude \"{str(input_fft_ampl_path)}\", shape={ampl.shape}, type={ampl.dtype} {ampl.nbytes/1073741824:0.3f} GB total. Took {t1-t0:0.2f} seconds" )
        
        avg_power_img = ampl[0]
        wanted_px_pctiles = [ 0, 2.5, 97.5, 100 ]
        px_pctiles = np.percentile( avg_power_img.flat, wanted_px_pctiles )
        pixmin, lbound, ubound, pixmax = px_pctiles#[ int(round(_)) for _ in px_pctiles ]
        np.clip( avg_power_img, lbound, ubound, out=avg_power_img )
        avg_power_img = ( ( avg_power_img - lbound ) * (255/(ubound-lbound)) ).round().astype( np.uint8 )
        imsave( str(output_avg_power_img_path), avg_power_img )
        
        if debug:
            t1 = time.time()
            print( f"Creating \"{str(output_avg_power_img_path)}\" took {t1-t0:0.2f} seconds" )

    # ========================================================
    # Part 2: Load amplitude (if not loaded already) and create histogram:            
    if output_fft_histogram_path.exists():
        print( f'Skipping creation of "{str(output_fft_histogram_path)}" since it already exists.')
    else:
        if debug:
            t0 = time.time()
        if ampl is None:
            ampl = np.load( str( input_fft_ampl_path ) )
    
            if debug:
                t1 = time.time()
                print( f"Loaded amplitude \"{str(input_fft_ampl_path)}\", shape={ampl.shape}, type={ampl.dtype} {ampl.nbytes/1073741824:0.3f} GB total. Took {t1-t0:0.2f} seconds" )

        deltat, framerate, timestamps = ReadTimestamps( input_timestampdata_path )
      
        def format_func( fft_bin, ticknumber ):
            #print (deltat, value, deltat/value)
            period = deltat / fft_bin
            freq = 1 / period
            return f"{freq:0.1f}"
        
        fig, (ax0, ax1) = plt.subplots( nrows=2, dpi=150, constrained_layout=True, figsize=(10,7))
        title = f'{input_vid_fname_stem}\nMean FFT magnitude vs. frequency\n'                 f'Total elapsed t={deltat:0.1f}s, {len(timestamps)} frames, {framerate:0.2f} ms/frame'
        fig.suptitle( title )
        
        Y0 = mean_amplitudes = CalculateMeanAmplitudes( ampl )
        X0 = np.arange( start=1, stop=len( Y0 ) )
        ax0.scatter( X0, Y0, marker='x' )
        ax0.set_ylabel( 'Mean magnitude (dB)' )
        ax0.set_xlabel( 'Frequency (Hz)')
        ax0.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        # Graph only top 100 in subfigure
        from operator import itemgetter
        X1, Y1 = zip( *list( reversed( sorted( zip( X0, Y0 ), key=itemgetter(1) ) ))[:100] )
        ax1.scatter( X1, Y1, marker='x' )
        ax1.set_xlabel( 'FFT bin')
        ax1.set_ylabel( 'Mean magnitude (dB)' )
        
        def bot2top_scale( fft_bin ):
            period = deltat / fft_bin
            freq = 1 / period
            return freq

        def top2bot_scale( fft_bin ):
            return fft_bin # pass
        
        secax = ax1.secondary_xaxis('top', functions=(bot2top_scale, top2bot_scale))
        #secax.set_xlabel('Frequency (Hz)')
        
        fig.savefig( str(output_fft_histogram_path) )
        plt.close(fig) # Explicitly close to free memory and avoid warning

        if debug:
            t1 = time.time()
            print( f"Creating \"{str(output_fft_histogram_path)}\" took {t1-t0:0.2f} seconds" )
            
    # ========================================================
    # Part 3: Load amplitude (if not loaded already) and create PCA pixel similarity:            
    if output_pca_path.exists():
        print( f'Skipping creation of "{str(output_pca_path)}" since it already exists.')
    else:
        if debug:
            t0 = time.time()
        if ampl is None:
            ampl = np.load( str( input_fft_ampl_path ) )
    
            if debug:
                t1 = time.time()
                print( f"Loaded amplitude \"{str(input_fft_ampl_path)}\", shape={ampl.shape}, type={ampl.dtype} {ampl.nbytes/1073741824:0.3f} GB total. Took {t1-t0:0.2f} seconds" )

        if debug:
            t0 = time.time()
        # drop average power bin (the 0th index of the 0th axis)
        ampl = np.delete( ampl, obj=0, axis=0 )
        # make sure there are no references to ampl so that when we delete it, the memory is freed
        from copy import deepcopy
        old_shape = deepcopy( ampl.shape )
        nframes, nrows, ncols = old_shape
        # Reshape pixel intensities into 2D array
        # Put all the pixels on one axis
        ampl_reshaped = ampl.reshape( ( len(ampl), -1 ) )
        ampl_reshaped = ampl_reshaped.transpose()
        if debug:
            t1 = time.time()
            print( f"Reshaping amplitude array from {old_shape} to {ampl_reshaped.shape} took {t1-t0:0.2f} seconds" )
        del ampl

        if debug:
            t0 = time.time()
        from sklearn.decomposition import PCA
        n_components = 5
        model = PCA( n_components=n_components )
        model.fit( ampl_reshaped )
        evr = model.explained_variance_ratio_
        ampl_PCA = model.transform( ampl_reshaped )
        del ampl_reshaped
        ampl_PCA = ampl_PCA.reshape( ( nrows, ncols, n_components ) )
        ampl_PCA = np.moveaxis( ampl_PCA, 2, 0 )
        
        fig, axes = plt.subplots( nrows=n_components, dpi=600, figsize=(8, 10), constrained_layout=True)
        title = f'{input_vid_fname_stem}\nPixel similarity as determined by PCA\n'
        fig.suptitle( title )
        for i in range(n_components):
            axes[i].imshow( ampl_PCA[i] )
            axes[i].set_axis_off()
            axes[i].set_title( f"PC{i} (explained variance={evr[i]*100:0.2f}%)" )

        fig.savefig( str( output_pca_path ) )
        plt.close(fig) # Explicitly close to free memory and avoid warning

        if debug:
            t1 = time.time()
            print( f"PCA model fit (evr={evr}) and writing {str( output_pca_path )} took {t1-t0:0.2f} seconds" )
    # ========================================================
    # Part 4: ISOCHRONS
    if output_isochron_subdir.exists():
        print( f'Skipping creation of isochronic mapping since "{str(output_isochron_subdir)}" already exists.')
    else:
        output_isochron_subdir.mkdir()
        if debug:
            t0 = time.time()
        if ampl is None:
            ampl = np.load( str( input_fft_ampl_path ) )
    
            if debug:
                t1 = time.time()
                print( f"Loaded amplitude \"{str(input_fft_ampl_path)}\", shape={ampl.shape}, type={ampl.dtype} {ampl.nbytes/1073741824:0.3f} GB total. Took {t1-t0:0.2f} seconds" )
        # Load Phase from disk
        if debug:
            t0 = time.time()
        phase = np.load( str( input_fft_phase_path ) )
        if debug:
            t1 = time.time()
            print( f"Loaded phase \"{str(input_fft_phase_path)}\", shape={phase.shape}, type={phase.dtype} {phase.nbytes/1073741824:0.3f} GB total. Took {t1-t0:0.2f} seconds" )
        
        if debug:
            t0 = time.time()
            
        n_isochrons = 50

        if deltat is None:
            deltat, framerate, timestamps = ReadTimestamps( input_timestampdata_path )
            
        if mean_amplitudes is None:
            mean_amplitudes = CalculateMeanAmplitudes( ampl )

        Y0 = mean_amplitudes
        X0 = np.arange( start=1, stop=len( Y0 ) )
        from operator import itemgetter
        mean_ampl_iter = reversed( sorted( zip( X0, Y0 ), key=itemgetter(1) ) )

        for rank, (fft_bin, mean_ampl) in enumerate( mean_ampl_iter, start=1 ):
            freq = fft_bin/deltat
            kwargs = {
                "amplitudes" : ampl[ fft_bin ],
                "phases" : phase[ fft_bin ],
                "rank" : rank,
                "frequency" : freq,
                "fft_bin" : fft_bin,
                "output_figure_path_prefix" : output_isochron_subdir / \
                    (f"{input_vid_fname_stem}_ISOCHRON_bin" + str(fft_bin).zfill(3) +\
                    f"_freq={freq:0.2f}" + "_rank" + str(rank).zfill(3))
            }
            GenerateIsochronFigure( **kwargs )
            if rank == n_isochrons:
                break

        if debug:
            t1 = time.time()
            print( f"Wrote {n_isochrons} isochronic color maps to {str( output_isochron_subdir )}; took {t1-t0:0.2f} seconds" )

    print( f"TOTAL PROCESSING TIME for \"{input_vid_fname_stem}\" took {time.time()-t00:0.2f} seconds.")


# In[12]:


get_ipython().system('rm -rfv /home/colettace/projects/rostislav/pipeline_output/isochronic_color_maps/*')


# In[13]:


for paths in input_data:
    FFT_PostProcessing( *paths )


# In[ ]:




