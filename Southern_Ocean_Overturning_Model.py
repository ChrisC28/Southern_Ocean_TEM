# -*- coding: utf-8 -*-
#==============================================================================#
# Southern_Ocean_Overturning_Model
#
# This script executes the conceptual model of Southern Ocean described in
# Chapman & Sallée, originally introduced by Marshall & Radko (2003) and Marshall & Radko (2006).
# The model solves the Transformed Eulerian Mean equations:
#
# [tau_wind/f + Psi_res] d(b)/dz + Kd(b)/dy = 0
#
# by the method of characteristics. In lieu of solving the PDE directly, we 
# the coupled system of ODEs
# dz/ds = [tau_wind/f + Psi_res] 
# dy/ds = K
# db/ds = 0
#
# Here, tau_wind, is the wind stress, and K is the eddy diffusivity, both set by the user.
# The user must specify the form of the eddy diffusivity and the profiles of the 
# wind stress.
#  
# Psi_res is the residual overturning which is determined as part of the solution by the boundary 
# conditions. 
#
# Boundary conditions are Dirichlet at the surface:
# b(y,z=0) = g(y)
# and on the northern boundary y=Ly
# b(y=Ly,z) = f(y)
# both of which are set by the user. 
# The value of Psi_res is unknown at the start. It is taken to be a free-parameter
# which is determined in order to satisfy  the boundary conditions. We use here
# the shooting method to determine the value of the overturning streamfunction. 
# 
# The results are written to numpy format files with a naming convention
#
#==============================================================================#
# DEPENDANCIES
# This code requires numpy and the scipy interpolation toolbox to run.
# both are Open Source and availble free of charge from the websites below.
# http://www.numpy.org/
# https://www.scipy.org/ 
#==============================================================================#
#CONTACT
# This code was written by Chris Chapman. 
# Questions, commments and bugs can be sent to: 
# chris.chapman.28@gmail.com
#==============================================================================#
#
#==============================================================================#
#LICENCE 
#Copyright (C) 26/7/2016 by Christopher Chapman 
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
#to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
#and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
#OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
#USE OR OTHER DEALINGS IN THE SOFTWARE.
##========================================================================================##
import numpy as np
from scipy.interpolate import  interp1d,interp2d



def Interp_To_Regular_Grid(y_points,z_points,point_values,y_grid,z_grid,search_radius_y,search_radius_z):
    #==========================================================================#
    # Function interpolates the output obtained from the method of 
    # characteristics to a regular grid using a simple "casting the net" approach
    # 
    values_on_grid = np.zeros([z_grid.size,y_grid.size],dtype='float64')
    
   
    for iZ in range(0,z_grid.size):
        for iY in range(0,y_grid.size):
            
            dist_y = y_points-y_grid[iY]
            dist_z = z_points-z_grid[iZ]
            #Get all points within the Y and Z search radius

            indicies_to_get = np.nonzero(np.logical_and(np.abs(dist_y)<search_radius_y,
                                                        np.abs(dist_z)<search_radius_z))
            #Average them, ignoring NaN values
            values_on_grid[iZ,iY] = np.nanmean(point_values[indicies_to_get])
    
    return values_on_grid

def Int_RK4(state_vec,equation_rhs,time, time_step,params):
    #==========================================================================#
    #Int_RK4
    #
    #Integrate a system of ODEs with the form:
    # x_dot = RHS(x,time,parameters)
    #forward in time using a RK4 algorithm
    #equation_rhs is a function object that takes as input the state vector 
    #and the parameter vetor
    #
    #INPUTS
    # state_vec    - and N dimension state vector
    # equation_rhs - function object that is the right hand side of the ODE
    #                returns an N dimensional array as output
    # time         - time vector
    # time_step    - the delta_t used to step the equations forward
    # params       - a vector of input parameters for the RHS 
    #
    # RETURNS
    # N-dimensional vector of x_n+1 
    one_on_6 = 1.0/6.0
    
    F1 = equation_rhs(state_vec,time,params)
    F2 = equation_rhs(state_vec+(0.5*time_step*F1),time+0.5*time_step,params)
    F3 = equation_rhs(state_vec + (0.5*time_step*F2),time+0.5*time_step,params)
    
    F4 = equation_rhs(state_vec+(time_step*F3),time+time_step,params)
    
    return state_vec + (time_step*one_on_6) * (F1 + 2.0*F2 + 2.0*F3  + F4 )

def equation_system(state_vec,dist,parameters):
    #The TEM equations to be integrated. 
    #This function is used as input to RK4
    
    #Unpack 
    wind_stress = parameters[0]
    f           = parameters[1]
    K           = parameters[2] 
    y_lat       = parameters[3]
    z_grid      = parameters[4]
    
    z = state_vec[0]
    y = state_vec[1]
    b = state_vec[2]
    psi_res = state_vec[3]

    
    current_wind_stress = np.interp(y, y_lat, wind_stress)
    
    #interp_K  = interp2d(y_lat, -z_grid, K, kind='linear')
    current_K  = K(y,z)
    
    dzds = ( (current_wind_stress/f) + psi_res)/current_K
    #dzds =  -np.sqrt(( (-current_wind_stress/f) - psi_res)/K)
    dyds = 1.0
    dbds = 0
    dpsi_resds = 0

    return np.asarray([dzds,dyds,dbds,dpsi_resds])

def Solve_TEM_System(surface_bcs,step_size,max_n_steps,max_y,max_depth,param_vector,eqn_system,integrator,return_output=False):
    #==========================================================================#
    # Solve the TEM equation set using the method of characteristics and an RK4 
    # integrator
    
    #Initialize the state vector with the surface boundary conditions
    state_vector = np.zeros(4,dtype='float64')
    
    state_vector[0] = surface_bcs[0]
    state_vector[1] = surface_bcs[1]
    state_vector[2] = surface_bcs[2]
    state_vector[3] = surface_bcs[3]

    #Possibly part of the equations... include for completeness
    dist_on_isopycnal = 0
    
    if return_output:
        save_frequency = 10
        z  = np.zeros([max_n_steps/save_frequency],dtype='float64')
        y  = np.zeros([max_n_steps/save_frequency],dtype='float64')
        b  = np.zeros([max_n_steps/save_frequency],dtype='float64')
        psi_res = np.zeros([max_n_steps/save_frequency],dtype='float64')
    
    
    counter=0
    for i_step in range(1,max_n_steps):
        
        dist_on_isopycnal = dist_on_isopycnal+step_size
        state_vector =  integrator(state_vector,eqn_system,dist_on_isopycnal,step_size,param_vector) 
        if return_output and (0==i_step % save_frequency):
        
            z[counter]       = state_vector[0]
            y[counter]       = state_vector[1]
            b[counter]       = state_vector[2]
            psi_res[counter] = state_vector[3]
            counter = counter+1
                
                        
        if (state_vector[0]<-max_depth) or (state_vector[1]>max_y):
            if return_output: 
                return [z,y,b,psi_res]
            else:
                return state_vector
 
    if return_output:   
        return [z,y,b,psi_res]
    else:
        return state_vector

#==============================================================================#
# Output file paths and file names
#==============================================================================#

output_file_path = '/home/cchlod/ARGO_Analysis/Mapped_Output/TEM_Model_Output_2/'
output_file_name_stem = 'TEM_Overturning_Var_K_'

#==============================================================================#
# Initialise the model grid
#==============================================================================#
Ly = 2000e3  #y-domain length
Dz = 4000    #z depth

Lx = 21000e3 #x-domain length ... only used for specifying the value of psi_rs
delta_y = 10e3 #y grid spacing
delta_z = 10   #z grid spacing

f0 = -1.0e-4 #Corriolis parameter

y_grid = np.arange(0,Ly+delta_y,delta_y)
z_grid = np.arange(0,Dz+delta_z,delta_z)

#==============================================================================#
# Error parameters
#==============================================================================#
                                       
ERR_TOL  = 5.0 #isopycnal depth error at the northern boundary in metres 
MAX_ITER = 100 #maximum number of iterations before we give up trying to satisfy
               #the boundary conditions
#==============================================================================#
# Boundary conditions and parameters MUST BE SET IN ADVANCE
#==============================================================================#

#surface buoyancy distribution
DELTA_B       = 7.0e-3                                                          #buoyoancy gain across the ACC 
surface_b     = DELTA_B*y_grid/y_grid[-1]                                       #linear buoyancy profile


#northern buoyancy profile
#Here, we use an exponential profile a la Marshall & Radko (2006) - see their equation 14

h_scale =1000.0                                                                 #e-folding scale                                    
b_profile_north  = surface_b[-1] * (np.exp(-z_grid/h_scale))
interp_function_b_north = interp1d(b_profile_north,-z_grid,bounds_error=False,fill_value=np.nan)

#Surface wind stress
WIND_STRESS_0 = 2.0e-4                                                          #Max wind stress magnitude
wind_stress   = WIND_STRESS_0*(0.3 + np.sin(np.pi*y_grid/y_grid[-1]))           #Standard sinusoidal profile

#Eddy diffusivity
#The eddy difusivity has a vertical structure, a Guassian centred on the 
# critical layer depth, with a constant background diffusivity K_diff_0 and a peak 
# diffusivity of K_mod, and a standard deviation of diff_z_scale
# See Eqn. (31) of Chapman & Sallée

K_diff_0       = 250                                                            #Background Diffusivity
K_mod          = np.arange(500,3501,250)                                        #Peak diffusivity
critical_layer_z = np.arange(2000,2001,250)                                     #Critical Layer Depth
diff_z_scale   = 500.0                                                          #diffusivity depth scale

#==============================================================================#
# Solver and output parameters parameters.
#==============================================================================#
delta_s = 100.0                                                                 #integation grid spacing
n_steps  = 250000                                                               #max number of integration steps
save_frequency = 10                                                             #save the output every ? integration steps 

YY,ZZ = np.meshgrid(y_grid,z_grid)
for i_critical_layer in range(0,critical_layer_z.size):
    for iK in range(0,K_mod.size):
        #Start the integration for critical layer depth i_critical_layer and 
        #peak diffusivity iK
        #Writing 
        output_file_name = output_file_name_stem + str(K_mod[iK]) + '_zc_' + str(critical_layer_z[i_critical_layer]) + '.npz'
        print '==================================================================='
        print output_file_name
        
        #Set the eddy diffusivity (Eqn. 31 in Chapman & Sallée
        K_diff = np.zeros([z_grid.size,y_grid.size],dtype='float64') + K_diff_0
        K_diff = K_diff + K_mod[iK]*np.exp( -np.power((-ZZ-(-critical_layer_z[i_critical_layer])),2.0)/(2*diff_z_scale*diff_z_scale)  )
        interp_K  = interp2d(y_grid, -z_grid, K_diff, kind='linear')

        #Allocation of arrays for the storage of the model output
        z  = np.zeros([y_grid.size,n_steps/save_frequency],dtype='float64')
        y  = np.zeros([y_grid.size,n_steps/save_frequency],dtype='float64')
        b  = np.zeros([y_grid.size,n_steps/save_frequency],dtype='float64')
        psi_res = np.zeros([y_grid.size,n_steps/save_frequency],dtype='float64')

        #Set up the parameter vector to be passed to the model routines to solve
        #the coupled set of ODEs
         
        params = [wind_stress,f0,interp_K,y_grid,z_grid]
        max_depth = 3900.0

        #Integrate along the isopycnal that outcrops with the surface at y = y[iY] 
        for iY in range(0,y_grid.size):

            for i_iter in range(0,MAX_ITER):
                counter = 0
                if i_iter==0:
                    #initialise the overturning guess and the error vector for 
                    #each guess of the overturing 
                    psi_guess = [0.0,0.0]
                    z_error   = [0.0,0.0]
                    
                    #initial condition for the ODE
                    surface_BCs = [0,y_grid[iY],surface_b[iY],psi_guess[0]]
    
                    #integrate the TEM equations from y=y[iY] to the northern b'dy.
                    #using Psi_res = 0 as an initial guess
                    solution_at_end = Solve_TEM_System(surface_BCs,delta_s,n_steps,y_grid[-1],max_depth,params,equation_system,Int_RK4,return_output=False)
    
                    #What is the depth and the buoyancy of the isopycnal at the 
                    #northern boundary???
                    z_model_at_end     = solution_at_end[0]
                    b_model_at_end     = solution_at_end[2]

                    #Find what the depth of the isopycnal _should_ be according to the 
                    #northern boundary conditions
                    z_equivalent = interp_function_b_north(b_model_at_end)
                    #and determine the error
                    z_error[0]      = z_equivalent - z_model_at_end
    
                    if z_error[0] > 0:
                        #Too deep! Try a stronger (positive) overturning 
                        psi_guess[1] =  40*1.0e6/Lx
                    elif z_error[0] < 0:
                        #Too shallow! Try a stronger (negative) overturning 
                        psi_guess[1] = -40*1.0e6/Lx

                    #Resolve the TEM equations using the new value of Psi_res
                    
                    surface_BCs = [0,y_grid[iY],surface_b[iY],psi_guess[1]]
                    solution_at_end = Solve_TEM_System(surface_BCs,delta_s,n_steps,y_grid[-1],max_depth,params,equation_system,Int_RK4,return_output=False)
                    z_model_at_end     = solution_at_end[0]
                    b_model_at_end = solution_at_end[2]
    
                    #Calculate the error with the new value of Psi_res
                    z_equivalent = interp_function_b_north(b_model_at_end)
                    z_error[1]      = z_equivalent - z_model_at_end
        
                if np.isnan(z_error[0]) or np.isnan(z_error[1]):
                    #Oh no! No convergence on this isopycnal... happens sometime
                    #at the far south of the domain 
                    print 'No convergence at: ', i_iter
                    break     
                
                #If the errors are of opposite sign, we can use the bisection 
                #method to refine the guess of Psi_res systematically
                if z_error[0]*z_error[1]<0:
            
                    new_psi_guess = 0.5*(psi_guess[0]+psi_guess[1])             #New guess of Psi_res... half way 
                                                                                #between the two previous guesses
                    #Resolve the TEM equations with the new Psi guess
                    surface_BCs = [0,y_grid[iY],surface_b[iY],new_psi_guess]
    
                    solution_at_end = Solve_TEM_System(surface_BCs,delta_s,n_steps,y_grid[-1],max_depth,params,equation_system,Int_RK4)
                    z_model_at_end     = solution_at_end[0]
                    b_model_at_end     = solution_at_end[2]

                    #Get the equivalent z 
                    z_equivalent = interp_function_b_north(b_model_at_end)
        
                    new_z_error = z_equivalent - z_model_at_end   
                    #Have we reached convergence or have we reached the maximum
                    #number of iterations? If so, stop here
                    if np.abs(new_z_error) <ERR_TOL or (i_iter==MAX_ITER-1):
                        
                        psi_res[iY,0] = 0.5*(psi_guess[0]+psi_guess[1])         #Final guess of Psi_res
                        
                        #Solve the system, this time using the final guess of Psi_res
                        #and saving the entire array
                        output_solution = Solve_TEM_System(surface_BCs,delta_s,n_steps,y_grid[-1],
                                                   max_depth,params,equation_system,Int_RK4,True)
                        z[iY,:]       = output_solution[0]
                        y[iY,:]       = output_solution[1]
                        b[iY,:]       = output_solution[2]
                        psi_res[iY,:] = output_solution[3]
 
                        print 'Converged after ', i_iter, ' iterations' 
                        break
                    
                    #Not converged and errors are of same sign? 
                    #Shuffle to get the new guess and iterate again
                    #This is the bisection method
                    elif new_z_error*z_error[0] >0:
                        psi_guess[0] = new_psi_guess
                        z_error[0]   = new_z_error
                    #Not converged and errors are of opposite sign? 
                    #Shuffle to get the new guess and iterate again
                    else:
                        psi_guess[1] = new_psi_guess
                        z_error[1]   = new_z_error
                
                #initial guess of Psi_res are of the same sign? 
                #Add or subtract 10Sv to the Psi_res guess and see if we can't 
                #improve the error
                else:
                    #Find the best guess from the previous Psi_res
                    new_psi_guess = psi_guess[np.argmin(np.abs(z_error))]
                    #Too shallow.... increase the overturning to flatten isopycnals
                    if z_error[np.argmin(np.abs(z_error))]<0: 
                        new_psi_guess = new_psi_guess - (10.0e6/Lx)
                    #Too deep.... increase the overturning to flatten isopycnals
                    elif z_error[np.argmin(np.abs(z_error))]>0:                          
                        new_psi_guess = new_psi_guess + (10.0e6/Lx)
                    
                    #Resolve the TEM equations with the new guess for Psi_res
                    surface_BCs = [0,y_grid[iY],surface_b[iY],new_psi_guess]
         
                    solution_at_end = Solve_TEM_System(surface_BCs,delta_s,n_steps,y_grid[-1],max_depth,params,equation_system,Int_RK4)
                    z_model_at_end     = solution_at_end[0]
                    b_model_at_end     = solution_at_end[2]

                    #Get the equivalent z and the new error
                    z_equivalent = interp_function_b_north(b_model_at_end)
                    new_z_error = z_equivalent - z_model_at_end   
                    if (new_z_error==z_error[0]) or (new_z_error==z_error[1]):
                        print 'No convergence at: ', i_iter     
                        break
                    
                    #Iterate again and try to improve the guess of Psi_res
                    psi_guess[0] = psi_guess[np.argmin(np.abs(z_error))]
                    psi_guess[1] = new_psi_guess
                
                    z_error[0]   = z_error[np.argmin(np.abs(z_error))]
                    z_error[1]   = new_z_error
         
        #The output of the TEM model is on a very unstructured set of points.
        #Interpolate to a regular grid before saving the output   
        b_on_grid       = Interp_To_Regular_Grid(y[:,::5].flatten(),z[:,::5].flatten(),b[:,::5].flatten(),y_grid,-z_grid,100.0e3,20.0)
        psi_res_on_grid = Interp_To_Regular_Grid(y[:,::5].flatten(),z[:,::5].flatten(),psi_res[:,::5].flatten(),y_grid,-z_grid,100.0e3,20.0)

        #Save the output
        np.savez(output_file_path+output_file_name, b_on_grid,psi_res_on_grid,y_grid,z_grid)
    
    #END iK in range(0,Kmod.size)
#END i_critical_layer
