#required modules
from itertools import count
import os
from tkinter import image_names
import numpy as np
from random import randrange, uniform
from matplotlib import pyplot as plt
import math
import matplotlib.image as mpimg
import random
##############################################
class create_round_path():
    #create traj from 0 to 2 or otherwise
    def create_half_circle_0_2(self,radius,points,S,road_size):
        choose=random.choice([1,-1])
        if S==0:
            if choose==1:
                radius+=road_size
            else:
                radius-=road_size
        else:
            if choose==1:
                radius-=road_size
            else:
                radius+=road_size

        angle = np.linspace( 0 , choose * np.pi , points ) 
        x = radius * np.cos( angle ) 
        y = radius * np.sin( angle ) 
        return x.tolist(),y.tolist()
    ###############################################################################
    #create traj from 1 to 3 or otherwise
    def create_half_circle_1_3(self,radius,points,S,road_size):
        choose=random.choice([1,-1])
        if S==1:
            if choose==1:
                radius-=road_size
            else:
                radius+=road_size
        else:
            if choose==1:
                radius+=road_size
            else:
                radius-=road_size
        angle = np.linspace( 0 , choose * np.pi , points ) 
        x = radius * np.sin( angle ) 
        y = radius * np.cos( angle ) 
        return x.tolist(),y.tolist()
    ###############################################################################
    #create traj from 2 to 3 or otherwise
    def create_quart_circle_2_3(self,radius,points,S,road_size):
        choose=0.5
        if S==2:
            radius-=road_size
        else:
            radius+=road_size
        angle = np.linspace( 0 , choose * np.pi , points ) 
        x = radius * np.cos( angle ) 
        y = radius * np.sin( angle ) 
        return x.tolist(),y.tolist()
    ###############################################################################
    #create traj from 1 to 2 or otherwise
    def create_quart_circle_1_2(self,radius,points,S,road_size):
        choose=-0.5
        if S==1:
            radius -=road_size
        else:
            radius +=road_size
        angle = np.linspace( 0 , choose * np.pi , points ) 
        x = radius * np.cos( angle ) 
        y = radius * np.sin( angle ) 
        return x.tolist(),y.tolist()
    ###############################################################################
    #create traj from 0 to 3 or otherwise
    def create_quart_circle_0_3(self,radius,points,S,road_size):
        choose=-0.5
        if S==0:
            radius +=road_size
        else:
            radius -=road_size
        angle = np.linspace( 0 , choose * np.pi , points ) 
        x = radius * np.sin( angle ) 
        y = radius * np.cos( angle ) 
        return x.tolist(),y.tolist()
    ###############################################################################
    #create traj from 0 to 1 or otherwise
    def create_quart_circle_0_1(self,radius,points,S,road_size):
        choose=0.5
        if S==0:
            radius-=road_size
        else:
            radius+=road_size
        angle = np.linspace( 0 , choose* np.pi , 150 ) 
        x = -1*radius * np.cos( angle ) 
        y = -1*radius * np.sin( angle ) 
        
        return x.tolist(),y.tolist()
    ###############################################################################
    #create traj path based on source and destination index
    def create_round_path(self,source,destination,points,radius,road_size):
        L=[source,destination]
        if (source+destination)==4:
            #verical round intersection
            points=int(points/2)
            x,y =self.create_half_circle_1_3(radius,points,source,road_size)
            
        elif (source+destination)==2:
            #horizantal round intersection
            points=int(points/2)
            x,y =self.create_half_circle_0_2(radius,points,source,road_size)
            
        elif (source+destination)==5:
            #quarter circle between path 2,3
            points=int(points/4)
            x,y =self.create_quart_circle_2_3(radius,points,source,road_size)
            
        elif (source+destination)==1:
            #quarter circle between path 0,1
            points=int(points/4)
            x,y= self.create_quart_circle_0_1(radius,points,source,road_size)
            
        elif (source+destination)==3 and ( L==[0,3] or L[::-1]==[0,3]):
            #quarter circle between path 0,3
            points=int(points/4)
            x,y= self.create_quart_circle_0_3(radius,points,source,road_size)
            
        elif (source+destination)==3 and ( L==[1,2] or L[::-1]==[1,2]):
            #quarter circle between path 1,2
            points=int(points/4)
            x,y= self.create_quart_circle_1_2(radius,points,source,road_size)
        return x,y

class create_linear_path():
    def create_linear_path_0(self,step,rang,radius,S_OR_D,road_size,is_round):
        X,Y=[],[]
        #check if path contain round
        if is_round==False:  
            if S_OR_D=='source':
                value=road_size
            else:
                value=-road_size
            
            for i in range(0,rang,step):
                X.append(-i)
                Y.append(value)
            return X,Y
        else:
            if S_OR_D=='source':
                Y_point=-road_size
            else:
                Y_point=road_size
            X_point=math.sqrt((radius-road_size)**2-Y_point**2)
            for i in range(0,rang+road_size,step):
                X.append(-(X_point+i))
                Y.append(Y_point)
            return X,Y
            
            
    def create_linear_path_1(self,step,rang,radius,S_OR_D,road_size,is_round):
        X,Y=[],[]
        #check if path contain round
        if is_round==False:
            if S_OR_D=='source':
                value=-road_size
            else:
                value=road_size
            for i in range(0,rang,step):
                X.append(value)
                Y.append(-i)
            return X,Y
        else:
            if S_OR_D=='source':
                X_point=-road_size
            else:
                X_point=road_size
            Y_point=math.sqrt((radius-road_size)**2-X_point**2)
            for i in range(0,rang+road_size,step):
                X.append(X_point)
                Y.append(-(Y_point+i))
            return X,Y

    def create_linear_path_2(self,step,rang,radius,S_OR_D,road_size,is_round):
        #check if path contain round
        X,Y=[],[]
        if is_round==False:
            if S_OR_D=='source':
                value=-road_size
            else:
                value=road_size
            for i in range(0,rang,step):
                X.append(i)
                Y.append(value)
            return X,Y
        else:
           
            if S_OR_D=='source':
                Y_point=-road_size
            else:
                Y_point=road_size
            X_point=math.sqrt((radius-road_size)**2-Y_point**2)
            for i in range(0,rang+road_size,step):
                X.append(X_point+i)
                Y.append(Y_point)
            return X,Y
    
    def create_linear_path_3(self,step,rang,radius,S_OR_D,road_size,is_round):
        #check if path contain round
        X,Y=[],[]
        if is_round==False:
            if S_OR_D=='source':
                value=road_size
            else:
                value=-road_size
            for i in range(0,rang,step):
                X.append(value)
                Y.append(i)
            return X,Y
        else:
            
            if S_OR_D=='source':
                X_point=road_size
            else:
                X_point=-road_size
            Y_point=math.sqrt((radius-road_size)**2-X_point**2)
            for i in range(0,rang+road_size,step):
                X.append(X_point)
                Y.append(Y_point+i)
            return X,Y

    def create_path(self,path_number,step,rang,radius,S_OR_D,road_size,is_round):
        if path_number==0:
            x,y=self.create_linear_path_0(step,rang,radius,S_OR_D,road_size,is_round)
            return x,y
        elif path_number==1:
            x,y=self.create_linear_path_1(step,rang,radius,S_OR_D,road_size,is_round)
            return x,y
        elif path_number==2:
            x,y=self.create_linear_path_2(step,rang,radius,S_OR_D,road_size,is_round)
            return x,y
        elif path_number==3:
            x,y=self.create_linear_path_3(step,rang,radius,S_OR_D,road_size,is_round)
            return x,y

class create_path(create_linear_path,create_round_path):
    #create od matrix randomaly
    def create_od_matrix(self,raw_size):
        OD_MATRIX=[randrange(10000) for i in range(raw_size*raw_size)]
        if raw_size==3:
            #no return
            OD_MATRIX[0],OD_MATRIX[4],OD_MATRIX[8]=0,0,0
        elif raw_size==4:
            #no return
            OD_MATRIX[0],OD_MATRIX[5],OD_MATRIX[10],OD_MATRIX[15]=0,0,0,0
        elif raw_size==2:
            OD_MATRIX[0],OD_MATRIX[3]=0,0
        OD_MATRIX=np.array(OD_MATRIX).reshape(raw_size,raw_size).tolist()
        return OD_MATRIX
    ##############################################################################################
    #obtain source , destination index based on od matrix
    def source_destination_prob(self,OD_MATRIX,raw_size):
        #get source index
        source_path_sum=[sum(OD_MATRIX[i]) for i in range(raw_size)]
        total_sum=sum(source_path_sum)
        source_path_prob=[source_path_sum[i]/total_sum for i in range(raw_size)]
        choose=random.choices([i for i in range(raw_size)],weights=source_path_prob)
        choose=choose[0]
        #get destination index

        OD_MATRIX=np.array(OD_MATRIX).T.tolist()
        destination_path_sum=[]
        for i in range(raw_size):
            if i !=choose:
                destination_path_sum.append(sum(OD_MATRIX[i]))
            else:
                destination_path_sum.append(0)
        total_sum_destination=sum(destination_path_sum)
        destination_path_prob=[destination_path_sum[i]/total_sum_destination for i in range(raw_size)]
        choose_des=random.choices([i for i in range(raw_size)],weights=destination_path_prob)
        choose_des=choose_des[0]
        return choose,choose_des
    #sampling data based on sampling rate
    def sampling_function(self,x,y,sampling_rate):
        #sampling the data so we don`t take whole data
        sampled_x=[]
        sampled_y=[]
        total_index=len(x)
        index_list=[i for i in range(total_index)]
        random_sampling=random.choices(index_list,k=int(total_index*sampling_rate)  )
        for i in random_sampling:
            sampled_x.append(x[i])
            sampled_y.append(y[i])
        return sampled_x,sampled_y
    ###############################################################################################
    #adding noise to traj data
    def adding_noise(self,x,y,noise_size_x,noise_size_y,mu,sigma_x,sigma_y):
        
        
        x=np.array(x)
        y=np.array(y)
        noise_x = np.random.normal(mu,sigma_x, x.shape)
        noise_y = np.random.normal(mu,sigma_y, y.shape)
        new_x = x + noise_size_x * noise_x
        new_y= y + noise_size_y * noise_y
        new_x=new_x.tolist()
        new_y=new_y.tolist()
        return new_x,new_y
    ###############################################################################################
    #create whole traj between source and destination index
    def total_path(self,source,destination,step,rang,radius,is_round,points,road_size): 
        if is_round==False:
            
            x_1,y_1=self.create_path(source,step,rang,radius,'source',road_size,False)
            x_2,y_2=self.create_path(destination,step,rang,radius,'destination',road_size,False)
            x=x_1+x_2
            y=y_1+y_2

        else: 

            x_1,y_1=self.create_path(source,step,rang,radius,'source',road_size,True)
            x_2,y_2=self.create_path(destination,step,rang,radius,'destination',road_size,True)
            x_3,y_3=self.create_round_path(source,destination,points,radius,road_size)
            x=x_1+x_2+x_3
            y=y_1+y_2+y_3
        return x,y
    
class create_and_plotting(create_path):
    def plotting_data_with_saving(self,x, y, fig_count, path,x_min,x_max,y_min,y_max):
        plt.axis('off')
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.scatter(x,y , alpha= 0.8, c='black', s=15)
        plt.savefig(path+'//'+str(fig_count)+'.jpg')
        plt.clf()
    ############################################################################################
    #create cross intersection
    def cross_intersection(self,number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,image_check):
        raw_size=4
        radius=0
        is_round=False
        points=0
        X,Y=[],[]
        
        OD_MATRIX=self.create_od_matrix(raw_size)
        for i in range(int(number_of_trips*sampling_trip)):
            source,destination=self.source_destination_prob(OD_MATRIX,raw_size)
            x,y=self.total_path(source,destination,step,rang,radius,is_round,points,road_size)
            x,y=self.sampling_function(x,y,sampling_rate)
            x,y=self.adding_noise(x,y,noise_size_x,noise_size_y,mu,sigma_x,sigma_y)
            X+=x
            Y+=y
        self.plotting_data_with_saving(X,Y,fig_count,path,x_min,x_max,y_min,y_max)
        if image_check==True:
            self.check_to_remove(path,fig_count)
    ############################################################################################
    #create t intersection
    def T_intersection(self,number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,fixed_shape,image_check):
        is_round=False
        radius=0
        points=0
        X,Y=[],[]
        if fixed_shape==True:
            raw_size=3
            OD_MATRIX=self.create_od_matrix(raw_size)
        else :
            raw_size=4
            OD_MATRIX=self.create_od_matrix(raw_size)
            canceled_raw=random.randint(0,3)
            for i in range(raw_size):
                OD_MATRIX[canceled_raw][i]=0
                OD_MATRIX[i][canceled_raw]=0
        for i in range(int(number_of_trips*sampling_trip)):
            source,destination=self.source_destination_prob(OD_MATRIX,raw_size)
            x,y=self.total_path(source,destination,step,rang,radius,is_round,points,road_size)
            x,y=self.sampling_function(x,y,sampling_rate)
            x,y=self.adding_noise(x,y,noise_size_x,noise_size_y,mu,sigma_x,sigma_y)
            X+=x
            Y+=y
        self.plotting_data_with_saving(X,Y,fig_count,path,x_min,x_max,y_min,y_max)
        if image_check==True:
            self.check_to_remove(path,fig_count)
    ############################################################################################
    #create round intersection
    def round_intersection(self,number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,points,image_check):
        is_round=True
        X,Y=[],[]
        raw_size=4
        OD_MATRIX=self.create_od_matrix(raw_size)
        for i in range(int(number_of_trips*sampling_trip)):
            source,destination=self.source_destination_prob(OD_MATRIX,raw_size)
            x,y=self.total_path(source,destination,step,rang,radius,is_round,points,road_size)
            x,y=self.sampling_function(x,y,sampling_rate)
            x,y=self.adding_noise(x,y,noise_size_x,noise_size_y,mu,sigma_x,sigma_y)
            X+=x
            Y+=y
        self.plotting_data_with_saving(X,Y,fig_count,path,x_min,x_max,y_min,y_max)
        if image_check==True:
            self.check_to_remove(path,fig_count)
    ############################################################################################
    #create non intersection intersection
    def non_intersection(self,number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,fixed_shape,with_nodes,image_check,strait_prob=0.5):
        is_round=False
        X,Y=[],[]
        points=0
        radius=0
        if with_nodes==True:
            strait=[[0,2],[1,3]]
            inter=[[0,1],[2,3],[0,3],[1,2]]
            total=[strait,inter]
            w_1=[strait_prob,1-strait_prob]
            w_2=[0.5,0.5]
            w_3=[0.25,0.25,0.25,0.25]
            type_inter=random.choices(total,weights=w_1)[0]
            if len(type_inter)==4:
                S_AND_D=random.choices(inter,weights=w_3)[0]
                source=S_AND_D[0]
                destination=S_AND_D[1]
            else:
                S_AND_D=random.choices(strait,weights=w_2)[0]
                source=S_AND_D[0]
                destination=S_AND_D[1]
        else:
            if fixed_shape==True:
                source=0
                destination=2
            else:
                strait=[[0,2],[1,3]]
                w_1=[0.5,0.5]
                S_AND_D=random.choices(strait,weights=w_1)[0]
                source=S_AND_D[0]
                destination=S_AND_D[1]
        
        
        for i in range(int(number_of_trips*sampling_trip)):
            
            x_1,y_1=self.total_path(source,destination,step,rang,radius,is_round,points,road_size)
            x_2,y_2=self.total_path(destination,source,step,rang,radius,is_round,points,road_size)
            x=x_1+x_2
            y=y_1+y_2
            x,y=self.sampling_function(x,y,sampling_rate)
            x,y=self.adding_noise(x,y,noise_size_x,noise_size_y,mu,sigma_x,sigma_y)
            X+=x
            Y+=y
        self.plotting_data_with_saving(X,Y,fig_count,path,x_min,x_max,y_min,y_max)
        if image_check==True:
            self.check_to_remove(path,fig_count)
    def check_to_remove(self,path,fig_count):
        img = mpimg.imread(path+'/'+str(fig_count)+'.jpg')
        imgplot = plt.imshow(img)
        plt.show()
        
        if input('you want to save it yes ,no ?  ')=='no':
            os.remove(path+'//'+str(fig_count)+'.jpg')
        plt.clf()

class manual_executing(create_and_plotting):
    def executing_cross(self):
        number_of_trips=int(input('Enter number of trips in the intersection (inter number) ? : '))
        rang=int(input('Enter the length of the road (intger number ) ? : '))
        step=int(input('Enter the length between each samples (intger number) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1) ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (range between 1 and 10) ? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (range between 1 and 10) ? : '))
        mu=int(input('Enter the mean for add noise ? : '))
        sigma_x=int(input('Enter standard deviation for x label ? : '))
        sigma_y=int(input('Enter standard deviation for y label ? : '))
        sampling_trip=float(input('Enter sampling rate for trips (float between 0,1) ? : '  ))
        road_size=int(input('Enter the distance between the two neiboaring roads ? : '))
        road_size=road_size//2
        fig_count=int(input('Enter fig number to save (intger number) ? : '))
        path=input('Enter path to save the image like (E://akram//fahmy/) ? : ')
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        check=input('want to check image before save it (true,false) ? : ')
        if check=='true':
            image_check=True
        else:
            image_check=False
        radius=0
        self.cross_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,image_check)


    def executing_T(self):
        number_of_trips=int(input('Enter number of trips in the intersection (inter number) ? : '))
        rang=int(input('Enter the length of the road (intger number ) ? : '))
        step=int(input('Enter the length between each samples (intger number) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1) ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (range between 1 and 10) ? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (range between 1 and 10) ? : '))
        mu=int(input('Enter the mean for add noise ? : '))
        sigma_x=int(input('Enter standard deviation for x label ? : '))
        sigma_y=int(input('Enter standard deviation for y label ? : '))
        sampling_trip=float(input('Enter sampling rate for trips (float between 0,1) ? : '  ))
        road_size=int(input('Enter the distance between the two neiboaring roads ? : '))
        road_size=road_size//2
        fig_count=int(input('Enter fig number to save (intger number) ? : '))
        path=input('Enter path to save the image like (E://akram//fahmy/)? : ')
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        check=input('want to check image before save it (true,false) ? : ')
        if check=='true':
            image_check=True
        else:
            image_check=False
        radius=0
        check=input('want fixed shape (true,false) ? : ')
        if check=='true':
            fixed_shape=True
        else:
            fixed_shape=False
        self.T_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,fixed_shape,image_check)


    def executing_round(self):
        number_of_trips=int(input('Enter number of trips in the intersection (inter number) ? : '))
        rang=int(input('Enter the length of the road (intger number ) ? : '))
        step=int(input('Enter the length between each samples (intger number) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1) ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (range between 1 and 10) ? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (range between 1 and 10) ? : '))
        mu=int(input('Enter the mean for add noise ? : '))
        sigma_x=int(input('Enter standard deviation for x label ? : '))
        sigma_y=int(input('Enter standard deviation for y label ? : '))
        sampling_trip=float(input('Enter sampling rate for trips (float between 0,1) ? : '  ))
        road_size=int(input('Enter the distance between the two neiboaring roads ? : '))
        road_size=road_size//2
        fig_count=int(input('Enter fig number to save (intger number) ? :'))
        path=input('Enter path to save the image like (E://akram//fahmy/) ? : ')
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        check=input('want to check image before save it (true,false) ? : ')
        if check=='true':
            image_check=True
        else:
            image_check=False
        radius=int(input('Enter radius of round circle (intger bigger than distance between two beside road) ? : '))
        points=int(input('Enter number of points to represent a round circle (intger number)? : '))
        
        self.round_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,points,image_check)

    def executing_nointersection(self):
        number_of_trips=int(input('Enter number of trips in the intersection (inter number) ? : '))
        rang=int(input('Enter the length of the road (intger number ) ? : '))
        step=int(input('Enter the length between each samples (intger number) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1 ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (range between 1 and 10) ? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (range between 1 and 10) ? : '))
        mu=int(input('Enter the mean for add noise ? : '))
        sigma_x=int(input('Enter standard deviation for x label ? : '))
        sigma_y=int(input('Enter standard deviation for y label ? : '))
        sampling_trip=float(input('Enter sampling rate for trips (float between 0,1) ? : '  ))
        road_size=int(input('Enter the distance between the two neiboaring roads ? : '))
        road_size=road_size//2
        fig_count=int(input('Enter fig number to save (intger number) ? : '))
        path=input('Enter path to save the image like (E://akram//fahmy/) ? : ')
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        check=input('want to check image before save it (true,false) ? : ')
        if check=='true':
            image_check=True
        else:
            image_check=False
        radius=0
        check=input('want fixed shape (true,false) ? : ')
        if check=='true':
            fixed_shape=True
        else:
            fixed_shape=False
        check=input('want nodes  (true,false) ? : ')
        if check=='true':
            with_nodes=True
        else:
            with_nodes=False
        
        strait_prob=float(int('Enter straight line prob (float between 0,1) ? : '))
        self.non_intersection(self,number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,fixed_shape,with_nodes,image_check,strait_prob)

class automated_executing(create_and_plotting):
    def default_values_cross(self,fig_count,path,required_count):
        count=0
        photo_name=fig_count
        generated_flag=False
        while count<=required_count:
            number_of_trips = random.randint(50, 101)
            rang = random.randint(40, 101)
            x_min=-rang
            x_max=rang
            y_min=-rang
            y_max=rang
            step = random.randint(1, 5)
            sampling_rate = random.random()
            noise_size_x = random.randint(1, 6) 
            noise_size_y = random.randint(1,6)
            mu = 0
            image_check=False
            sigma_x = uniform(0.5,2)
            sigma_y=uniform(0.5,2)
            sampling_trip= random.random()
            road_size = random.randint(1, 4)
            radius = 0
            points = 0
            
            
            if (sampling_rate + sampling_trip > 0.5) and (sampling_rate > 0.1) and (sampling_trip > 0.1) and (noise_size_x + noise_size_y < 9):
                self.cross_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,image_check)
                generated_flag=True
            if(generated_flag):
                count+=1
                photo_name+=1
            
            generated_flag=False

    def default_values_T(self,fig_count,path,required_count):
        count=0
        photo_name=fig_count
        generated_flag=False
        while count<=required_count:
            number_of_trips = random.randint(50, 101)
            rang = random.randint(40, 101)
            x_min=-rang
            x_max=rang
            y_min=-rang
            y_max=rang
            step = random.randint(1, 5)
            sampling_rate = random.random()
            noise_size_x = random.randint(1, 6) 
            noise_size_y = random.randint(1,6)
            mu = 0
            image_check=False
            sigma_x = uniform(0.5,2)
            sigma_y=uniform(0.5,2)
            sampling_trip= random.random()
            road_size = random.randint(1, 4)
            radius = 0
            points = 0
            
            fixed_shape=False
            if (sampling_rate + sampling_trip > 0.5) and (sampling_rate > 0.1) and (sampling_trip > 0.1) and (noise_size_x + noise_size_y < 9):
               
                self.T_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,fixed_shape,image_check) 
                generated_flag=True
            if(generated_flag):
                count+=1
                photo_name+=1
            generated_flag=False     
    
    def default_values_round(self,fig_count,path,required_count):
        count=0
        photo_name=fig_count
        generated_flag=False
        while count<=required_count:
            number_of_trips = random.randint(50, 101)
            rang = random.randint(40, 101)
            
            step = random.randint(1, 5)
            sampling_rate = random.random()
            noise_size_x = random.randint(1, 6) 
            noise_size_y = random.randint(1,6)
            mu = 0
            image_check=False
            sigma_x = uniform(0.5,2)
            sigma_y=uniform(0.5,2)
            sampling_trip= random.random()
            road_size = random.randint(1, 4)
            radius = random.randint(40, 71)
            points = random.randint(40, 81)
            x_min=-rang-radius
            x_max=rang+radius
            y_min=-rang-radius
            y_max=rang+radius
            
            if (sampling_rate + sampling_trip > 0.5) and (sampling_rate > 0.1) and (sampling_trip > 0.1) and (noise_size_x + noise_size_y < 9):
                self.round_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,points,image_check)
                generated_flag=True
            if(generated_flag):
                count+=1
                photo_name+=1
            generated_flag=False

    def default_values_nointersection(self,fig_count,path,required_count):
        count=0

        generated_flag=False
        photo_name=fig_count
        while count<=required_count:
            number_of_trips = random.randint(50, 101)
            rang = random.randint(40, 101)
            x_min=-rang
            x_max=rang
            y_min=-rang
            y_max=rang
            step = random.randint(1, 5)
            sampling_rate = random.random()
            noise_size_x = random.randint(1, 6) 
            noise_size_y = random.randint(1,6)
            mu = 0
            image_check=False
            sigma_x = uniform(0.5,2)
            sigma_y=uniform(0.5,2)
            sampling_trip= random.random()
            road_size = 0
            radius = 0
            points = 0
            
            fixed_shape=False
            with_nodes=False
            strait_prob=0.5
            if (sampling_rate + sampling_trip > 0.5) and (sampling_rate > 0.1) and (sampling_trip > 0.1) and (noise_size_x + noise_size_y < 9):
                self.non_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,fixed_shape,with_nodes,image_check,strait_prob)
                generated_flag=True
            if(generated_flag):
                count+=1
                photo_name+=1
            generated_flag=False


        def automated_executing_cross(self,required_count):
            number_of_trips=int(input('Enter number of trips in the intersection (inter number) ? '))
            rang=int(input('Enter the length of the road (intger number ) ? '))
            step=int(input('Enter the length between each samples (intger number) ? '))
            sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1 ? ' ))
            noise_size_x=int(input('Enter how far noise is present in x label for data (range between 1 and 10)? '))
            noise_size_y=int(input('Enter how far noise is present in y label for data (range between 1 and 10)? '))
            mu=0
            sigma_x=1.2
            sigma_y=1.2
            sampling_trip=1
            road_size=int(input('Enter the distance between the two neiboaring roads ? '))
            road_size=road_size//2
            fig_count=int(input('Enter base fig number to save (intger number)'))
            path=input('Enter path to save the image like (E://akram//fahmy/)? ')
            x_min=float(input('Enter  minimum  range for plotting in x label ? '))
            x_max=float(input('Enter  maximum  range for plotting in x label ? '))
            y_min=float(input('Enter  minimum  range for plotting in y label ? '))
            y_max=float(input('Enter  maximum  range for plotting in y label ? '))
            image_check=False
            radius=0
            count=0
            photo_name=fig_count
            while photo_name <count:
                
                self.cross_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,fig_count,path,x_min,x_max,y_min,y_max,image_check)

    def range_values_T(self,fig_count,path,required_count):
        
        number_of_trips=int(input('Enter number of trips in the intersection (bigger than 20) ? : '))
        rang=int(input('Enter the length of the road (bigger than 10 ) ? : '))
        step=int(input('Enter the length between each samples (less than 40) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1) ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (less than 3) ? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (less than 3) ? : '))
        mu=0
        sigma_x=1
        sigma_y=1
        sampling_trip=1
        road_size=int(input('Enter the distance between the two neiboaring roads (less than 100) ? : '))
        road_size=road_size//2
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        image_check=False
        fixed_shape=input('fixed shape (true or false) ? : ')
        if fixed_shape=='true':
            fixed_shape=True
        else:
            fixed_shape=False
        radius=0
        ##################3
        count=0
        photo_name=fig_count
        
        while count<=required_count:
            number_of_trips = random.randint(20,number_of_trips)
            rang = random.randint(10,rang)
            x_min=-rang
            x_max=rang
            y_min=-rang
            y_max=rang
            step = random.randint(1,step)
            
            noise_size_x = random.randint(noise_size_x,3) 
            noise_size_y = random.randint(noise_size_y,3)
            
            
           
            
            road_size = random.randint(0,road_size)
            radius = 0
            points = 0
            
            
            
            
            self.T_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,fixed_shape,image_check) 
            count+=1
            photo_name+=1
            
    def range_values_cross(self,fig_count,path,required_count):
        
        number_of_trips=int(input('Enter number of trips in the intersection (bigger than 20) ? : '))
        rang=int(input('Enter the length of the road (bigger than 10 ) ? : '))
        step=int(input('Enter the length between each samples (less than 40) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1 ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (less than 5) ? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (less than 5) ? : '))
        mu=0
        sigma_x=1
        sigma_y=1
        sampling_trip=1
        road_size=int(input('Enter the distance between the two neiboaring roads less than 100 ? : '))
        road_size=road_size//2
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        image_check=False
        radius=0
        ##################3
        count=0
        photo_name=fig_count
        
        while count<=required_count:
            number_of_trips = random.randint(20,number_of_trips)
            rang = random.randint(10,rang)
            x_min=-rang
            x_max=rang
            y_min=-rang
            y_max=rang
            step = random.randint(1,step)
            
            noise_size_x = random.randint(noise_size_x,3) 
            noise_size_y = random.randint(noise_size_y,3)
            
            
           
            
            road_size = random.randint(0,road_size)
            radius = 0
            points = 0
            
            
            
            self.cross_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,image_check)
            
            count+=1
            photo_name+=1

    def range_values_round(self,fig_count,path,required_count):
         


                
        number_of_trips=int(input('Enter number of trips in the intersection (bigger than 20) ? : '))
        rang=int(input('Enter the length of the road (bigger than 10 ) ? : '))
        step=int(input('Enter the length between each samples (less than 40) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1) ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (less than 3) ? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (less than 3) ? : '))
        mu=0
        sigma_x=1
        sigma_y=1
        sampling_trip=1
        road_size=int(input('Enter the distance between the two neiboaring roads (less than 100) ? : '))
        road_size=road_size//2
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        image_check=False
        radius =int(input('Enter radius of the round (less than 100 and bigger than 10) ? : '))
        points = int(input('Enter number of points to represent round (less than 600) ? : '))
        ##################3
        count=0
        photo_name=fig_count
        
        while count<=required_count:
            number_of_trips = random.randint(20,number_of_trips)
            rang = random.randint(10,rang)
            x_min=-rang
            x_max=rang
            y_min=-rang
            y_max=rang
            step = random.randint(1,step)
            radius=random.randint(10,radius)
            points=random.randint(points,600)
            noise_size_x = random.randint(noise_size_x,3) 
            noise_size_y = random.randint(noise_size_y,3)
            
            
           
            
            road_size = random.randint(0,road_size)
            
            
            
            
            self.round_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,points,image_check)
            count+=1
            photo_name+=1

    def range_values_nointersection(self,fig_count,path,required_count):
        number_of_trips=int(input('Enter number of trips in the intersection (bigger than 20) ? : '))
        rang=int(input('Enter the length of the road (bigger than 10 ) ? : '))
        step=int(input('Enter the length between each samples (less than 40) ? : '))
        sampling_rate=float(input('Enter sampling rate of the Data (float number between 0 and 1) ? : ' ))
        noise_size_x=int(input('Enter how far noise is present in x label for data (less than 3)? : '))
        noise_size_y=int(input('Enter how far noise is present in y label for data (less than 3)? : '))
        mu=0
        sigma_x=1
        sigma_y=1
        sampling_trip=1
        road_size=int(input('Enter the distance between the two neiboaring roads (less than 100) ? : '))
        road_size=road_size//2
        x_min=float(input('Enter  minimum  range for plotting in x label ? : '))
        x_max=float(input('Enter  maximum  range for plotting in x label ? : '))
        y_min=float(input('Enter  minimum  range for plotting in y label ? : '))
        y_max=float(input('Enter  maximum  range for plotting in y label ? : '))
        image_check=False
        fixed_shape=input('fixed shape (true or false) ? : ')
        if fixed_shape=='true':
            fixed_shape=True
        else:
            fixed_shape=False
        radius=0
        with_nodes=input('with nodes or not (true or false) ? : ')
        if with_nodes=='true':
            with_nodes=True
        else:
            with_nodes=False
        strait_prob=float(input('Enter prob of strait line (float between 0,1) ? : '))
        ##################3
        count=0
        photo_name=fig_count
        
        while count<=required_count:
            number_of_trips = random.randint(20,number_of_trips)
            rang = random.randint(10,rang)
            x_min=-rang
            x_max=rang
            y_min=-rang
            y_max=rang
            step = random.randint(1,step)
            
            noise_size_x = random.randint(noise_size_x,3) 
            noise_size_y = random.randint(noise_size_y,3)
            
            
           
            
            road_size = random.randint(0,road_size)
            radius = 0
            points = 0
            
            
            
            self.non_intersection(number_of_trips,radius,rang,step,sampling_rate,noise_size_x,noise_size_y,mu,sigma_x,sigma_y,sampling_trip,road_size,photo_name,path,x_min,x_max,y_min,y_max,fixed_shape,with_nodes,image_check,strait_prob)
            
            count+=1
            photo_name+=1
class executing(manual_executing,automated_executing):
    def execute(self):
        while True:
            manual_or_automated=input('manual or scripting or restart or exit ? : ')
            if manual_or_automated=='manual':
                print('Manual Executing Intersection Simulation')
                intersection_type=input('Enter intersection type cross or T or round or nointersection ? : ')
                if intersection_type=='cross':
                    self.executing_cross()
                elif intersection_type=='T':
                    self.executing_T()
                elif intersection_type=='round':
                    self.executing_round()
                elif intersection_type=='nointersection':
                    self.executing_nointersection()
                else:
                    print('Unkowing type')
                    continue

                print('Finish Executing Intersection Simulation')
            elif manual_or_automated=='scripting':
                use_default=input('Use default setting to execute or use input range  (default , range) ? : ')
                required_count=int(input('Enter number of images to generate using default setting ? : '))
                intersection_type=input('Enter intersection type cross or T or round or nointersection ? : ')
                fig_count=int(input('base number to name the images ? : '))
                path=input('path to save the images ? : ')
                if use_default=='default':
                
                    print('Automated Executing Intersection Simulation using default setting  ')

                    if intersection_type=='cross':
                        self.default_values_cross(fig_count,path,required_count)
                    
                    elif intersection_type=='T':
                        self.default_values_T(fig_count,path,required_count)
                    
                    elif intersection_type=='round':
                        self.default_values_round(fig_count,path,required_count)
                    elif intersection_type=='nointersection':
                        self.default_values_nointersection(fig_count,path,required_count)
                elif use_default=='range' :
                    if intersection_type=='cross':
                        self.range_values_cross(fig_count,path,required_count)
                    elif intersection_type=='T':
                        self.range_values_T(fig_count,path,required_count)
                    elif intersection_type=='round':
                        self.range_values_round(fig_count,path,required_count)
                    elif intersection_type=='nointersection':
                        self.range_values_nointersection(fig_count,path,required_count)
                    else:
                        continue
                else:
                    continue
            elif manual_or_automated=='restart':
                print('restarting ....')
                continue
            elif manual_or_automated=='exit':
                exit()
            else:
                print('Error')
                continue
    





