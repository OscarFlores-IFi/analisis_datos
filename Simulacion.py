class Optimizacion: 
    def crear_ventanas(data,n_ventana):
        import numpy as np
        n_data = len(data)
        dat_new = np.zeros((n_data-n_ventana+1,n_ventana))
        for k in np.arange(n_ventana):
            dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
        return dat_new

    def portafolio(x,u,p,rcom):
        x_1 = x;
        vp = x[0]+p*x[1] #Valor presente del portafolios
        x_1[0] = x[0]-p*u-rcom*p*abs(u) #Dinero disponible
        x_1[1] = x[1]+u #Acciones disponibles
        return vp,x_1
    
    
    #% Función para realizar la simulación del portafolio
    def portafolio_sim(precio,sit,Ud):
        import numpy as np
        from Simulacion import Optimizacion
        portafolio = Optimizacion.portafolio
        T = np.arange(len(precio))
            
        Vp = np.zeros(T.shape)
        X  = np.zeros((T.shape[0]+1,2)) 
        u = np.zeros(T.shape)
        X[0][0] = 10000
        rcom = 0.0025
        
        for t in T:
            
            u_max = np.floor(X[t][0]/((1+rcom)*precio[t])) # Numero maximo de la operacion
            u_min  = X[t][1] # Numero minimo de la operacion
            
            #AC (operacion matricial)
            if Ud[int(sit[t])]>0:
                u[t] = u_max*Ud[int(sit[t])]
            else:
                u[t] = u_min*Ud[int(sit[t])]
            
            Vp[t],X[t+1]=portafolio(X[t],u[t],precio[t],rcom)
        
    #    return T,Vp,X,u
        return Vp
    
    def portafolios_sim(data,sit,Ud):
        import numpy as np
        from Simulacion import Optimizacion
        portafolio_sim = Optimizacion.portafolio_sim
        
        Sim = np.zeros((len(data),len(sit[0])))
        for i in range(len(data)):
            Sim[i] = portafolio_sim(data[i].Close[-len(sit[0]):],sit[i],Ud)
            
        return(Sim)
    
    def simulacion(csv,ndias,model_close,Ud): 
        import numpy as np
        import pandas as pd
        from Simulacion import Optimizacion
        portafolios_sim = Optimizacion.portafolios_sim
        crear_ventanas = Optimizacion.crear_ventanas

        
        # Cargamos bases de datos en .csv
        data = []
        for i in csv: 
            data.append(pd.read_csv(i, index_col=0))
            
        # Creamos ventanas de tiempo
        vent = []
        for j in data: 
            ven = []
            for i in ndias:
                ven.append(crear_ventanas(j['Close'],i))  # IMPORTANTE!! Se asume que las bases de datos siempre recibiran el nombre de una columna 'Close'
            vent.append(ven)
    
        # Se estandarizan los datos
        cont = len(ndias)    
        norm = []
        for j in vent:
            for i in range(cont):
                j[i] = np.transpose((j[i].transpose()-j[i].mean(axis=1))/j[i].std(axis=1))
            norm.append(j)
            
        # Se clasifica la situación de los precios en cada cluster de k-means.
        clasif_close = []
        for norm in norm:
            tmp = []
            for i in range(cont):
                tmp.append(model_close[i].predict(norm[i]))
            clasif_close.append(tmp)   
            
        # Cortar la longitud de las clasificaciones para que tengan la misma longitud
        for j in clasif_close:
            for i in range(cont):
                j[i]=j[i][len(norm[0][i])-len(vent[0][-1]):]
            
        # Situación de cada t en T.
        sit = []
        for j in clasif_close:
            s1 = np.zeros(len(j[0]))
            for i in range(cont):
                s1 += j[i]*2**i
            sit.append(s1)
    
        # Simulamos
        Sim = portafolios_sim(data,sit,Ud)
        return(Sim)
    
    
    
    
    
    

class Graficos: 
    def portafolio(x,u,p,rcom):
        x_1 = x;
        vp = x[0]+p*x[1] #Valor presente del portafolios
        x_1[0] = x[0]-p*u-rcom*p*abs(u) #Dinero disponible
        x_1[1] = x[1]+u #Acciones disponibles
        return vp,x_1
    
    
    #% Función para realizar la simulación del portafolio
    def portafolio_sim(precio,sit,Ud):
        import numpy as np
        from Simulacion import Graficos
        portafolio = Graficos.portafolio
        T = np.arange(len(precio))
            
        Vp = np.zeros(T.shape)
        X  = np.zeros((T.shape[0]+1,2)) 
        u = np.zeros(T.shape)
        X[0][0] = 10000
        rcom = 0.0025
        
        for t in T:
            
            u_max = np.floor(X[t][0]/((1+rcom)*precio[t])) # Numero maximo de la operacion
            u_min  = X[t][1] # Numero minimo de la operacion
            
            #AC (operacion matricial)
            if Ud[int(sit[t])]>0:
                u[t] = u_max*Ud[int(sit[t])]
            else:
                u[t] = u_min*Ud[int(sit[t])]
            
            Vp[t],X[t+1]=portafolio(X[t],u[t],precio[t],rcom)
        
        return T,Vp,X,u
#        return Vp