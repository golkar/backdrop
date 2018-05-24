import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import copy, os, pickle, argparse, time
import george # main package used for generating GPs
from george.kernels import ExpSquaredKernel
from scipy import interpolate

# The folder where the GP samples will be saved.
sample_folder = 'gp-samples/'
cross_folder = 'gp-crosses/'

# Class for generating GP samples.
class generate_patch():
    """
    This class generates a GP and returns a tuple of (sample, sample size).
    Note that it is optimized for samples of size 1024, but can generate
    any size sample with a little tweaking of the parameters.
    
    'size' determines the size of the sample generated before upscaling.
            The final size of the sample is size * upscale_factor.
    
    'corr_length' is determines the rough diameter of the GP blobs
            and can only be specified for the base GP (i.e. cannot be
            simultaneously defined with a subpatch)
    
    'subpatch' = None:
            In this case, the generate function of the instance
            runs the George package with the size and corr_length
            specified.
          
    'subpatch' = generate_patch instance:
            In this case, the generate function fills a sample
            given by size unifromly with the given generate_patch
            instance.
            
    'envelope' = True multiplies the generated sample by a Gaussian
            such that it falls off to zero near the edges. (This is 
            required for the base (subpatch) generator to get a 
            smooth sample)
            
    'wrap' = True wrap the sample on a torus. That is identifies the sides
             of the sample near the edge. (Use this on the sampling instance,
             not on the subpatch instance.)
             
    'wrap_factor' = 1 +/- offset adjusts the toroidal wrapping so that the 
            sample remains uniform. To adjust this factor, turn test=True.
            Then, if there is a gap in the generated toroidal sample, 
            increase wrap_factor. If there is an overlap, reduce. Such
            that in the end it is uniform.
                    
    
    'test' = True, if turned on for the big GP (not subpatch) replaces the
            GP so we can adjust the wrap_factor to make sure everything is
            uniform.
    """
    
    def __init__(self, size, corr_length = None, upscale_factor = 1, subpatch = None,
                 envelope = True, wrap = False, wrap_factor = 1, test = False):
        
        assert type(subpatch)==generate_patch or subpatch==None,\
            "'subpatch' must be either None or a generate_patch class instance."
        assert subpatch==None or corr_length==None,\
            "'corr_length' can only be defined for the base instance, i.e."\
            +  " it cannot be specified with a subpatch instance simultaneaously."
            
        
        self.wrap_factor = wrap_factor; self.size = size
        self.corr_length = corr_length; self.subpatch = subpatch
        self.upscale_factor = upscale_factor; self.wrap = wrap
        self.test = test; self.envelope = envelope
        
        # Defining the boundary. This is used for torodial wrapping. 
        # The numbers are related to the size of the Gaussian envelope.
        if subpatch == None: 
            self.boundary = size / 5 
            # Effective size is the size that survives the Gaussian envelope.
            self.effective_size = size - 2 * self.boundary 
        else:
            self.boundary = subpatch.boundary * 4 * self.subpatch.upscale_factor
            
            
    # The sample generating function. 
    def generate(self):
        
        n_points = self.size
        
        if self.subpatch == None: # i.e. for base instance.
            
            # Defining the scale for George for having corr_length ~ blob diamater.
            scale = (self.corr_length/self.size)**2
            
            kernel = ExpSquaredKernel(scale, ndim=2)
            gp = george.GP(kernel, solver=george.HODLRSolver)
            
            # Creating a grid of points for George input.
            x_points = y_points = np.linspace(0, 1, n_points)
            xx, yy = np.meshgrid(x_points, y_points)
            indep = np.vstack((np.hstack(xx),np.hstack(yy))).T
            
            if self.test:
                patch = np.ones([n_points,n_points])/2
            else: 
                # Calling on George to create the samples.
                patch = gp.sample(indep).reshape((n_points,n_points))
            
            # Using interpolating to upscale the result if requested.
            if self.upscale_factor > 1:
                f = interpolate.interp2d(x_points, y_points, patch, kind='cubic')
                x_points = y_points = np.linspace(0, 1, np.int(n_points * self.upscale_factor))
                patch = f(x_points, y_points)
                
            # Creating and applying the Gaussian envelope. The coefficient
            # in the exp (in this case 23), determines how big the envelope is.
            if self.envelope:
                envelope = np.exp(-23*((x_points.reshape(-1,1)-0.5)**2 + (y_points.reshape(1,-1)-0.5)**2))
                patch = patch * envelope
            
            return patch, self.size * self.upscale_factor            
        
        else: # i.e. subpatch is another instance.
            
            # Initiating the sample
            n_points = self.size
            patch = np.zeros([n_points , n_points])
            
            # Defining the upscaled full subpatch size
            subpatch_size = np.int(self.subpatch.size * self.subpatch.upscale_factor)
            
            # Figuring out how many subpatches we need to cover the sample (size / effective subpatch size)
            subpatch_eff_size = np.int(self.subpatch.effective_size * self.subpatch.upscale_factor)
            ratio = n_points / subpatch_eff_size
            
            factor = 5000 if self.test else 1 # If testing for unifromity, sample a LOT of patches.
            
            # The location of where the subpatch smample is to be placed (locs gives the top left corner)
            locs = np.random.randint(0, n_points - subpatch_size, [np.int(6 * ratio**2 * factor),2])
            
            # Drawing the subpatch samples.
            if self.test == False:
                for loc_pair in locs:
                    patch[loc_pair[0]:loc_pair[0]+subpatch_size,
                          loc_pair[1]:loc_pair[1]+subpatch_size] +=  self.subpatch.generate()[0]    
            else:
                for loc_pair in locs:
                    patch[loc_pair[0],loc_pair[1]]+=0.1 #If testing, just put 0.1
            
            # Torodial wrapping
            if self.wrap == True:
                w = self.wrap_factor 
                patch[:np.int(w*self.boundary+0.5),:] += patch[-np.int(w*self.boundary+0.5):,:]
                patch[-np.int(w*self.boundary+0.5):,:] = patch[:np.int(w*self.boundary+0.5),:]
                patch[:,:np.int(w*self.boundary+0.5)] += patch[:,-np.int(w*self.boundary+0.5):]
                patch[:,-np.int(w*self.boundary+0.5):] = patch[:,:np.int(w*self.boundary+0.5)]                    
                
              
            # Upscaling using interpolation.
            if self.upscale_factor > 1.0:
                x_points = y_points = np.linspace(0, 1, n_points)
                f = interpolate.interp2d(x_points, y_points, patch, kind='cubic')
                x_points = y_points = np.linspace(0, 1, np.int(n_points * self.upscale_factor+0.55))
                patch = f(x_points, y_points)
                
            return patch, self.size * self.upscale_factor

# Helper function to decide an efficient patch/subpatch size. 
# Total size is 1024 here.
def gentex(cl, n = 1, test = False, start = 1):
    
    if cl >= 50:
        final = generate_patch(size = 128, corr_length = cl/8, upscale_factor = 8,
                                subpatch = None, test=test, envelope = False)
        scale = 8
    
    else:
    
        resc_cl = np.min([8, cl])
        scale = cl / resc_cl

        minipatch1 = generate_patch(size = 50, corr_length = resc_cl, upscale_factor = scale,
                                    subpatch = None, test=test)
        final = generate_patch(size = 1024, corr_length = None, upscale_factor = 1,
                            subpatch = minipatch1, wrap = True, wrap_factor = 1.246, test = test)
        
    print('Starting on job with cl {} and ps {} (using x{:.2f} upscaling). Generating {} samples starting from v{} ... '.format(
                                cl, 1024, scale, n, start))
           

    for i in range(start,n+start):
        
        now = time.time()
        sample, sz = final.generate()
        sample = (sample - sample.mean())/sample.std()
        if int(cl) == cl: cl = int(cl)
        file_name = sample_folder + 'final_' +'size-'+str(sz) + '_corr-'+str(cl)+'_v'+str(i)
        np.save(file_name, sample)

        el = np.int(time.time()-now)
        print('iteration: {}, time taken: {:02d}:{:02d}:{:02d}, saved in file: {}.npy.'.format(
                    i, el//3600, el%3600//60, el%60, file_name))
        
# Defining main for size = 1024.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cl", help="The correlation length.", type = float)
    parser.add_argument("-n", "--numsamples", help="How many samples to generate.", type = int)
    parser.add_argument("-t", "--test", help="How many samples to generate.", action="store_true")
    parser.add_argument("-s", "--start", help="First save number.", type = int, default = 1)
    args = parser.parse_args()
    gentex(args.cl, n = args.numsamples,
           test = args.test, start = args.start)


        
if __name__ == "__main__":
    main()

    

################# Code for creating crosses of GPs at different scales.  ############################



# Print which GP samples are present in the sample folder.
def look_for_GPs():
    list_of_files = os.listdir(sample_folder)
    size_n_corr = [file.split('size-')[1].split('_v')[0].split('_corr-')\
                       for file in list_of_files]
    sizes = sorted(list(set([float(el[0]) for el in size_n_corr])))
    corrs = sorted(list(set([float(el[1]) for el in size_n_corr])))
    sizes = [int(el) if int(el)==el else el for el in sizes]
    corrs = [int(el) if int(el)==el else el for el in corrs]
    print('The available sizes and correlations are: {} and {}.'.format(sizes, corrs))

# Loading base GP from file.
class GP_texture():
    
    def __init__(self, size, corr_length, num_samples):
        
        self.size = size
        self.corr_length = corr_length
        
        self.samples = [] #Dump all the samples of this GP here.
        num = 0
        for v in range(1, 1 + num_samples):
            file_name = sample_folder + 'final_size-%d_corr-%s_v%d.npy'%(size, str(corr_length), v)
            try:
                self.samples.append(np.load(file_name))
                num += 1
            except: pass
                
        self.num_samples = num
        
        print( 'Found {num} samples with correlation length {corr}.'.format(num = num, corr = corr_length))
        
# Loading a collection of GP's class
class GP_collection():
    
    def __init__(self, size, corr_lengths):
        
        self.size = size
        self.corr_lengths = corr_lengths
        self.num_corrs = len(corr_lengths)
        
        #Load a dict of correlation classes. The key is the correlation lenght.
        self.textures = {str(i):GP_texture(size,i,1024) for i in corr_lengths }
        
        print('-' * 80)
        
        #Check that the GPs are properly normalized. 
        self.check_norms()
        
        print('-' * 80)
        
        self.plot_samples()
    
    def check_norms(self):
        
        deviation = 0
        for gp in self.textures.values():
            for sample in gp.samples:
                deviation += np.abs(sample.mean()) + np.abs(1-sample.std())
        
        print('The deviation from having mean zero and std one is {}.'.format(deviation))
        
    def plot_samples(self, zoom = 1):
        
        #Plot the center crop with the size given by zoom.
        Range = self.size//zoom
        range_start = (self.size - Range)//2
        range_end = range_start + Range
        
        if zoom == 1:
            print('Plotting the {0}x{0} samples...'.format(self.size))
        else:
            print('Plotting the {0}x{0} crops of the {1}x{1} samples...'.format(Range, self.size))   
        
        print()
        num_cols = 3
        num_rows = int(self.num_corrs/num_cols + 0.999999)
    
        fig = plt.figure(figsize=(num_cols * 6,num_rows * 6.6))
        num = 1
        for key, crss in self.textures.items():
            sample_plot = fig.add_subplot(num_rows, num_cols, num)
            sample_plot.set_title(key)
            im = sample_plot.imshow(crss.samples[0][range_start:range_end,range_start:range_end],
                                    cmap = 'inferno', vmax = 4.5, vmin = -4.5)
            num += 1
        plt.show()
        
        
        
# Tool for creating a cross of GPs given by a list of list of correlations
class GP_cross():
    
    def __init__(self, size, corr_classes, power=None):
        
        
        assert type(corr_classes)==list,\
            "Parameter 'corr_classes' must be a list of correlation classes."
        
        num_corr_per_class = [len(clss) for clss in corr_classes]
        assert max(num_corr_per_class) == min(num_corr_per_class), \
            "Number of correlations per class must be the same for all classes."
            
        self.corr_class_num = len(corr_classes)
        if power == None:
            power = np.ones(self.corr_class_num).tolist()     
        assert type(power)==list ,\
            "Parameter 'power' must be a list of relative powers for the correlation classes."
        assert len(power)==self.corr_class_num ,\
            "Parameter 'power' must have length equal to the number of correlation classes."  
                    
        self.size = size
        self.corr_classes = corr_classes
        
        self.power = power
        self.corr_list = [i for j in corr_classes for i in j] #flatten the corr_class list
        collection = GP_collection(size, self.corr_list) #Load the collection
        
        print('\n','='*80, '\n')
        
        self.num_corr_per_class = num_corr_per_class[0]
        self.num_cross = self.num_corr_per_class **  self.corr_class_num
        
        #Setting the GP class name
        GP_class_names = ['-'.join([str(i) for i in cor_class]) for cor_class in corr_classes] 
        self.GP_name = 'GP_' + '+'.join(GP_class_names)
        
        #Checking no overlap between the classes:
        try: 
            self.check_init()
        except ValueError as err:
            raise ValueError(err)
        
        # Loading the class structure. (The loaded collection does not have structure)
        self.tex_classes = [{str(i):collection.textures[str(i)] for i in corr_class }\
                                 for corr_class in corr_classes]      
        
        # Fixes the number of samples per class.
        try: 
            self.check_numbers()
        except ValueError as err:
            raise ValueError(err)
            
        #Calling the function to generate the crosses.
        self.create_cross()
        
        print('Created {} cross sets with {} samples per class. \n'.format(
                self.num_cross, self.samples_per_cross))
        
        self.plot_samples()
    
    
    def check_init(self): #Checks for overlap between classes.
        for i in range(self.corr_class_num):
            for j in range(i+1, self.corr_class_num):
                if set(self.corr_classes[i]).intersection(self.corr_classes[j]) != set():
                    raise ValueError('Overlap between correlation sets not allowed.')
    
    # Fixes the number of samples per class.
    def check_numbers(self): 
        
        #First figures out the minimum number of samples for all correlations.
        samples_per_corr = min([min([tex.num_samples for tex in tex_class.values()]) \
                             for tex_class in self.tex_classes])
        # Then figures out how many samples we will have per cross.
        self.samples_per_cross = samples_per_corr // (self.num_corr_per_class **  (self.corr_class_num - 1))
        
        # Then discards the extras (e.g. not divisible or more in one corr than another)
        samples_per_corr = self.samples_per_cross * (self.num_corr_per_class **  (self.corr_class_num - 1))
        for i, tex_class in enumerate(self.tex_classes):
            for tex in tex_class.values():
                tex.num_samples = samples_per_corr
                tex.samples = tex.samples[:samples_per_corr]
            
    
    # Helper functions to divide the samples into the crosses. 
    def dec_to_base(self, n):
        nums = []
        for i in range(self.corr_class_num):
            n, r = divmod(n, self.num_corr_per_class)
            nums.append(r)
        return nums

    def base_to_dec(self, nums):
        n = 0
        for pwr, coeff in enumerate(nums):
            n += coeff * self.num_corr_per_class**pwr
        return n
    
    #Creating the crosses here.    
    def create_cross(self):
        
        self.crosses = {}
        self.labels = {}
        
        for i in range(self.num_cross):
            
            # This tells us which corrs in each class participate in cross number i.
            class_choices = self.dec_to_base(i)
            
            # Figuring out the keys and the names.
            class_keys = [list(self.tex_classes[j].keys())[class_choices[j]] for j in range(self.corr_class_num)]
            class_name = '-'.join([str(key) for key in sorted(class_keys, key = lambda x:float(x))])
            
            # Populating the self.labels dictionary
            self.labels[class_name] = i
            
            # Creating the crosses for each cross class.
            cross_list = []
            for j in range(self.samples_per_cross):      
                
                sample = np.zeros([self.size, self.size]) 
                
                # Loop over all the classes. The choice of which correlation per class is given above.
                for class_number in range(self.corr_class_num):
                    
                    # To figure out which samples participate in this cross class,
                    # we use the helper functions to figure out how many samples already used.
                    ccc = copy.deepcopy(class_choices)
                    del(ccc[class_number])
                    pos = self.base_to_dec(ccc)*self.samples_per_cross
                    
                    corr_sample = self.tex_classes[class_number][class_keys[class_number]].samples[pos+j]
                    sample += corr_sample * self.power[class_number]
                
                cross_list.append(sample)
            
            self.crosses[class_name] = cross_list
            
            # Populating the inverse label lookup
            self.label_lookup = {}
            for name, num in self.labels.items():
                self.label_lookup[num] = name
            
    
    def plot_samples(self, zoom = 1):
        
        Range = self.size//zoom
        range_start = (self.size - Range)//2
        range_end = range_start + Range
        
        if zoom == 1:
            print('Plotting the generated {0}x{0} samples...'.format(self.size))
        else:
            print('Plotting the {0}x{0} crops of the generated {1}x{1} samples...'.format(Range, self.size))
           
        print()
        num_cols = 3
        num_rows = int(self.num_cross/num_cols + 0.999999)
    
        fig = plt.figure(figsize=(num_cols * 6,num_rows * 6.6))
        num = 1
        for key, crss in self.crosses.items():
            sample_plot = fig.add_subplot(num_rows, num_cols, num)
            sample_plot.set_title(key)
            im = sample_plot.imshow(crss[0][range_start:range_end,range_start:range_end], cmap = 'inferno')
            num += 1
        plt.show()
        
    def write_to_file(self, type = np.float32):
        
        to_write = []
        # Loop order is such that we have one sample per cross at a time.
        # For future truncation purposes.
        
        for i in range(self.samples_per_cross):
            num = 0
            for key, cross in self.crosses.items():
                to_write.append([cross[i].astype(type), self.labels[key]])
        path = cross_folder + self.GP_name + '.gp'
        
        with open(path,'wb') as ds: 
            pickle.dump([to_write, self.label_lookup], ds)
        
        print('Written to file {}.'.format(path))
        