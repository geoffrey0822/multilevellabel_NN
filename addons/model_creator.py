import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np

def exportModel(net,fname):
    with open(fname,'w') as f:
        f.write('%s\n'%net.to_proto())
        f.close()

def addConvolution(input,prefix,ksize,noutput,pad=0,stride=1,lr_mults=(0,0)):
    conv=L.Convolution(input,name='conv%s'%prefix,
                            convolution_param={'kernel_size':ksize,'num_output':noutput,'pad':pad,'stride':stride},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    relu=L.ReLU(conv,name='relu%s'%prefix,in_place=True)
    return conv,relu

def addBatchNorm(input,net,name,lr_mults=(1,1)):
    net.tops[name]=L.BatchNorm(input)
    scale_name='%s_scale'%name
    net.tops[scale_name]=L.Scale(net.tops[name],
                                 bias_term=True,
                                 in_place=True,
                                 param=[{'lr_mult':lr_mults[0],'decay_mult':1},
                                        {'lr_mult':lr_mults[1],'decay_mult':0}])
    return net.tops[scale_name]

def addInception_v1(input,prefix,ksizes,noutputs,pads,lr_mults=(0,0)):
    ways=len(ksizes)
    convs=[]
    relus=[]
    connectors=[]
    cnames=[]
    
    i=0
    for way in range(ways):
        header='incep_%d_%s'%(way+1,prefix)
        ksize=ksizes[way]
        noutput1=noutputs[way][0]
        noutput2=noutputs[way][1]
        pad=pads[way]
        conv1=L.Convolution(input,
                            convolution_param={'kernel_size':1,'num_output':noutput1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu1=L.ReLU(conv1,name='%s/relu1x1'%(header),
                     in_place=True)
        conv2=L.Convolution(conv1,
                            convolution_param={'kernel_size':ksize,'num_output':noutput2,'pad':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu2=L.ReLU(conv2,name='%s/relu%dx%d'%(header,ksize,ksize),
                     in_place=True)
        cnames.append('%s/conv%dx%d'%(header,ksize,ksize))
        cnames.append('%s/conv1x1'%header)
        convs.append(conv1)
        convs.append(conv2)
        relus.append(relu1)
        relus.append(relu2)
        connectors.append(conv2)
    pool=L.Pooling(input,name='%s/pool'%header,
                   pooling_param={'pool':P.Pooling.AVE,'kernel_size':1,'stride':1})
    connectors.append(pool)
    concatLayer=L.Concat(*connectors,name='incep_%s/join'%prefix)
    print cnames
    return convs,relus,pool,concatLayer,cnames

def addInception_v1_2(input,net,prefix,ksizes,noutputs,pads,lr_mults=(0,0)):
    ways=len(ksizes)
    convs=[]
    relus=[]
    connectors=[]
    cnames=[]
    
    i=0
    for way in range(ways):
        header='%s_%d'%(prefix,way+1)
        ksize=ksizes[way]
        noutput1=noutputs[way][0]
        noutput2=noutputs[way][1]
        pad=pads[way]
        
        conv1_name='%s/conv1'%header
        relu1_name='%s/relu1'%header
        conv2_name='%s/conv2'%header
        relu2_name='%s/relu2'%header
        
        net.tops[conv1_name]=L.Convolution(input,
                            convolution_param={'kernel_size':1,'num_output':noutput1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu1_name]=L.ReLU(net.tops[conv1_name],
                     in_place=True)
        net.tops[conv2_name]=L.Convolution(net.tops[conv1_name],
                            convolution_param={'kernel_size':ksize,'num_output':noutput2,'pad':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu2_name]=L.ReLU(net.tops[conv2_name],
                     in_place=True)
        connectors.append(net.tops[conv2_name])
    pool_name='%s/pool'%header
    net.tops[pool_name]=L.Pooling(input,
                   pooling_param={'pool':P.Pooling.AVE,'kernel_size':3,'stride':1,'pad':1})
    connectors.append(net.tops[pool_name])
    concat_name='%s/join'%prefix
    net.tops[concat_name]=L.Concat(*connectors)
    return net.tops[concat_name]

def addInception_v3(input,prefix,ksizes,noutputs,pads,lr_mults=(0,0)):
    ways=len(ksizes)
    convs=[]
    relus=[]
    connectors=[]
    i=0
    for way in range(ways):
        header='incep_%d_%s'%(way+1,prefix)
        ksize=ksizes[way]
        noutput1=noutputs[way][0]
        noutput2=noutputs[way][1]
        pad=pads[way]
        conv1=L.Convolution(input,
                            convolution_param={'kernel_size':1,'num_output':noutput1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu1=L.ReLU(conv1,name='%s/relu1x1'%(header),
                     in_place=True)
        conv2=L.Convolution(conv1,
                            convolution_param={'kernel_h':ksize,'kernel_w':1,'num_output':noutput2,'pad_h':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu2=L.ReLU(conv2,name='%s/relu%dx1'%(header,ksize),
                     in_place=True)
        conv3=L.Convolution(conv2,
                            convolution_param={'kernel_w':ksize,'kernel_h':1,'num_output':noutput2,'pad_w':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu3=L.ReLU(conv3,name='%s/relu1x%d'%(header,ksize),
                     in_place=True)
        convs.append(conv1)
        convs.append(conv2)
        convs.append(conv3)
        relus.append(relu1)
        relus.append(relu2)
        relus.append(relu3)
        connectors.append(conv3)
    pool=L.Pooling(input,name='%s/pool'%header,
                   pooling_param={'pool':P.Pooling.AVE,'kernel_size':1,'stride':1})
    connectors.append(pool)
    concatLayer=L.Concat(*connectors,name='incep_%s/join'%prefix)
    return convs,relus,pool,concatLayer

def addInception_v3_2(input,net,prefix,ksizes,noutputs,pads,lr_mults=(0,0)):
    ways=len(ksizes)
    connectors=[]
    i=0
    for way in range(ways):
        header='%s_%d'%(prefix,way+1)
        ksize=ksizes[way]
        noutput1=noutputs[way][0]
        noutput2=noutputs[way][1]
        pad=pads[way]
        
        conv1_name='%s/conv1'%header
        relu1_name='%s/relu1'%header
        conv2_name='%s/conv2'%header
        relu2_name='%s/relu2'%header
        conv3_name='%s/conv3'%header
        relu3_name='%s/relu3'%header
        
        net.tops[conv1_name]=L.Convolution(input,
                            convolution_param={'kernel_size':1,'num_output':noutput1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu1_name]=L.ReLU(net.tops[conv1_name],
                     in_place=True)
        net.tops[conv2_name]=L.Convolution(net.tops[conv1_name],
                            convolution_param={'kernel_h':ksize,'kernel_w':1,'num_output':noutput2,'pad_h':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu2_name]=L.ReLU(net.tops[conv2_name],
                     in_place=True)
        net.tops[conv3_name]=L.Convolution(net.tops[conv2_name],
                            convolution_param={'kernel_w':ksize,'kernel_h':1,'num_output':noutput2,'pad_w':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu3_name]=L.ReLU(net.tops[conv3_name],
                     in_place=True)
        connectors.append(net.tops[conv3_name])
    
    pool=L.Pooling(input,name='%s/pool'%header,
                   pooling_param={'pool':P.Pooling.AVE,'kernel_size':3,'stride':1,'pad':1})
    connectors.append(pool)
    concat_name='%s/join'%prefix
    net.tops[concat_name]=L.Concat(*connectors)
    return net.tops[concat_name]

def addInceptionRes(input,prefix,ksizes,noutputs,pads,lr_mults=(0,0)):
    ways=len(ksizes)
    convs=[]
    relus=[]
    connectors=[]
    i=0
    poutput=0
    for noutput in noutputs:
        poutput+=noutput[1]
    pconv=L.Convolution(input,
                        convolution_param={'kernel_size':1,'num_output':poutput},
                        param=[{'lr_mult':1,'decay_mult':2},
                               {'lr_mult':1,'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    prelu=L.ReLU(pconv,name='%s/relu0'%prefix,
                     in_place=True)
    for way in range(ways):
        header='incep_%d_%s'%(way+1,prefix)
        ksize=ksizes[way]
        noutput1=noutputs[way][0]
        noutput2=noutputs[way][1]
        pad=pads[way]
        conv1=L.Convolution(pconv,
                            convolution_param={'kernel_size':1,'num_output':noutput1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu1=L.ReLU(conv1,name='%s/relu1x1'%(header),
                     in_place=True)
        conv2=L.Convolution(conv1,
                            convolution_param={'kernel_h':ksize,'kernel_w':1,'num_output':noutput2,'pad_h':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu2=L.ReLU(conv2,name='%s/relu%dx1'%(header,ksize),
                     in_place=True)
        conv3=L.Convolution(conv2,
                            convolution_param={'kernel_w':ksize,'kernel_h':1,'num_output':noutput2,'pad_w':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        relu3=L.ReLU(conv3,name='%s/relu1x%d'%(header,ksize),
                     in_place=True)
        convs.append(conv1)
        convs.append(conv2)
        convs.append(conv3)
        relus.append(relu1)
        relus.append(relu2)
        relus.append(relu3)
        connectors.append(conv3)
    concatLayer=L.Concat(*connectors,name='incep_%s/join'%prefix)
    resLayer=L.Eltwise(*[pconv,concatLayer],
                       eltwise_param={'operation':P.Eltwise.SUM,'coeff':[1,-1]})
    
    return convs,relus,concatLayer,resLayer,pconv,prelu

def addInceptionRes2(input,net,prefix,ksizes,noutputs,pads,lr_mults=(0,0)):
    ways=len(ksizes)
    connectors=[]
    i=0
    poutput=0
    for noutput in noutputs:
        poutput+=noutput[1]
        
    pconv_name='%s_pconv'%prefix
    prelu_name='%s_prelu'%prefix
    net.tops[pconv_name]=L.Convolution(input,
                        convolution_param={'kernel_size':1,'num_output':poutput},
                        param=[{'lr_mult':1,'decay_mult':2},
                               {'lr_mult':1,'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    net.tops[prelu_name]=L.ReLU(net.tops[pconv_name],
                     in_place=True)
    for way in range(ways):
        header='%s_%d'%(prefix,way+1)
        ksize=ksizes[way]
        noutput1=noutputs[way][0]
        noutput2=noutputs[way][1]
        pad=pads[way]
        conv1_name='%s/conv1'%header
        relu1_name='%s/relu1'%header
        conv2_name='%s/conv2'%header
        relu2_name='%s/relu2'%header
        conv3_name='%s/conv3'%header
        relu3_name='%s/relu3'%header
        
        net.tops[conv1_name]=L.Convolution(net.tops[pconv_name],
                            convolution_param={'kernel_size':1,'num_output':noutput1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu1_name]=L.ReLU(net.tops[conv1_name],
                     in_place=True)
        net.tops[conv2_name]=L.Convolution(net.tops[conv1_name],
                            convolution_param={'kernel_h':ksize,'kernel_w':1,'num_output':noutput2,'pad_h':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu2_name]=L.ReLU(net.tops[conv2_name],
                     in_place=True)
        net.tops[conv3_name]=L.Convolution(net.tops[conv2_name],
                            convolution_param={'kernel_w':ksize,'kernel_h':1,'num_output':noutput2,'pad_w':pad,'stride':1},
                            param=[{'lr_mult':lr_mults[0],'decay_mult':2},
                                   {'lr_mult':lr_mults[1],'decay_mult':0}],weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
        net.tops[relu3_name]=L.ReLU(net.tops[conv3_name],
                     in_place=True)
        connectors.append(net.tops[conv3_name])
    concat_name='%s/join'%prefix
    net.tops[concat_name]=L.Concat(*connectors)
    resLayer_name='%s/res'%prefix
    net.tops[resLayer_name]=L.Eltwise(*[net.tops[pconv_name],net.tops[concat_name]],
                       eltwise_param={'operation':P.Eltwise.SUM,'coeff':[1,-1]})
    return net.tops[resLayer_name]
    
def addResidual(partA,partBs,net,prefix,operation='SUM'):
    for top in partBs:
        print top.fn.params

def addPartialRegression(inputs,net,prefix,last_size,feature_dim=0,with_relu=True,dropout_ratio=0.7,lr_mult=(1,1),loss_weight=1):
    pool_name='%s_partialReg/pool'%prefix
    fc_name='%s_partialReg/fc'%prefix
    relu_name='%s_partialReg/relu'%prefix
    dropout_name='%s_partialReg/dropout'%prefix
    accuracy_name='%s_partialReg/accuracy'%prefix
    loss_name='%s_partialReg/loss'%prefix
    
    net.tops[pool_name]=L.Pooling(inputs[0],
                                  pooling_param={'pool':P.Pooling.AVE,'kernel_size':last_size,'stride':1},
                                  exclude=dict(stage='deploy'))
    
    if feature_dim>0:
        feature_name='%s_partialReg/featureVector'%prefix
        feature_relu='%s_partialReg/featureReLU'%prefix
        feature_dropout='%s_partialReg/featureDropout'%prefix
        net.tops[feature_name]=L.InnerProduct(net.tops[pool_name],
                                     inner_product_param={'num_output':feature_dim},
                                     param=[{'lr_mult':lr_mult[0],'decay_mult':2},
                                            {'lr_mult':lr_mult[1],'decay_mult':0}],
                                     weight_filler=dict(type='xavier'),
                                     bias_filler=dict(type='constant',value=0),
                                     exclude=dict(stage='deploy'))
        if with_relu:
            net.tops[feature_relu]=L.ReLU(net.tops[feature_name],in_place=True,
                                      exclude=dict(stage='deploy'))
        if dropout_ratio>0:
            net.tops[feature_dropout]=L.Dropout(net.tops[feature_name],
                                            dropout_param={'dropout_ratio':dropout_ratio},
                                            in_place=True,
                                            exclude=dict(stage='deploy'))
        net.tops[fc_name]=L.InnerProduct(net.tops[feature_name],
                                     param=[{'lr_mult':lr_mult[0],'decay_mult':2},
                                            {'lr_mult':lr_mult[1],'decay_mult':0}],
                                     weight_filler=dict(type='xavier'),
                                     bias_filler=dict(type='constant',value=0),
                                     exclude=dict(stage='deploy'))
    else:
        net.tops[fc_name]=L.InnerProduct(net.tops[pool_name],
                                     param=[{'lr_mult':lr_mult[0],'decay_mult':2},
                                            {'lr_mult':lr_mult[1],'decay_mult':0}],
                                     weight_filler=dict(type='xavier'),
                                     bias_filler=dict(type='constant',value=0),
                                     exclude=dict(stage='deploy'))
    
    if with_relu:
        net.tops[relu_name]=L.ReLU(net.tops[fc_name],
                                   in_place=True,
                                   exclude=dict(stage='deploy'))
    
    net.tops[accuracy_name]=L.Accuracy(*[net.tops[fc_name],inputs[1]],
                                       include=dict(stage='val'))
    net.tops[loss_name]=L.SoftmaxWithLoss(*[net.tops[fc_name],inputs[1]],
                                          loss_weight=loss_weight,
                                          exclude=dict(stage='deploy'))
    
def deeper_inception(input,net,prefix,ksizess,noutputss,padss,currentSize,version='v1',full_residual=False,lr_mults=(1,1),base_loss_weight=1,discount=1,scale_bn=False):
    depth=len(ksizess)
    outputLayer=input
    layer_lose_weight=base_loss_weight
    for d in range(depth):
        new_prefix='%s_dep%d'%(prefix,d)
        ksizes=ksizess[d]
        pads=padss[d]
        noutputs=noutputss[d]
        temp_total=0
        for num_out in noutputs:
            temp_total+=num_out[len(num_out)-1]
            
        norm_name='%s/bn'%new_prefix
        p_layer=outputLayer
        if version=='v1':
            outputLayer=addInception_v1_2(outputLayer,net, new_prefix, ksizes, noutputs, pads, lr_mults)
        elif version=='v3':
            outputLayer=addInception_v3_2(outputLayer,net, new_prefix, ksizes, noutputs, pads, lr_mults)
        else:
            outputLayer=addInceptionRes2(outputLayer,net, new_prefix, ksizes, noutputs, pads, lr_mults)
            
        if d!=depth-1:
            norm_layer=None
            if scale_bn:
                norm_layer=addBatchNorm(outputLayer, net, norm_name, lr_mults)
            else:
                net.tops[norm_name]=L.BatchNorm(outputLayer)
                norm_layer=net.tops[norm_name]
            outputLayer=norm_layer
            if full_residual and version !='v1' and version !='v3':
                residual_name='%s/residual'%new_prefix
                net.tops[residual_name]=L.Eltwise(*[p_layer,outputLayer],
                                                  eltwise_param={'operation':P.Eltwise.SUM,'coeff':[1,-1]})
                outputLayer=net.tops[residual_name]
                
            addPartialRegression([outputLayer,net.label], net, new_prefix, currentSize,loss_weight=layer_lose_weight,feature_dim=temp_total)
            layer_lose_weight*=discount
    return outputLayer
    
def low_res_model(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='train-data',
                    include=dict(stage='train'),
                    ntop=2)
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,64,lr_mults=(1,1))
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 32x32
    
    iconcat=addInception_v1_2(net.pool1,net, 'incep1', ksizes=[7,5,3], noutputs=[[32,64],[32,64],[32,64]], pads=[3,2,1], lr_mults=(1,1))
    
    net.bn1=L.BatchNorm(iconcat)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    iconcat=addInception_v3_2(net.pool2,net, 'incep2', ksizes=[9,7,5,3], noutputs=[[64,96],[64,96],[64,96],[64,96]], pads=[4,3,2,1], lr_mults=(1,1))
    
    net.bn2=L.BatchNorm(iconcat)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 8x8
    iconcat=addInception_v3_2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[96,128],[96,128]], pads=[2,1], lr_mults=(1,1))
    
    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':4,'stride':4})
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    net.i1part_pool=L.Pooling(net.bn1,
                              pooling_param={'pool':P.Pooling.MAX,'kernel_size':16,'stride':16},
                              exclude=dict(stage='deploy'))
    net.i1part_fc=L.InnerProduct(net.i1part_pool,
                                 param=[{'lr_mult':1,'decay_mult':2},
                                        {'lr_mult':1,'decay_mult':0}],
                                 weight_filler=dict(type='xavier'),
                                 bias_filler=dict(type='constant',value=0),
                                 exclude=dict(stage='deploy'))
    net.i1part_softmaxLoss=L.SoftmaxWithLoss(*[net.i1part_fc,net.label],
                                             exclude=dict(stage='deploy'))
    net.i1part_accuracy=L.Accuracy(*[net.i1part_fc,net.label],
                            include=dict(stage='val'))
    
    net.i2part_pool=L.Pooling(net.bn2,
                              pooling_param={'pool':P.Pooling.MAX,'kernel_size':16,'stride':16},
                              exclude=dict(stage='deploy'))
    net.i2part_fc=L.InnerProduct(net.i2part_pool,
                                 param=[{'lr_mult':1,'decay_mult':2},
                                        {'lr_mult':1,'decay_mult':0}],
                                 weight_filler=dict(type='xavier'),
                                 bias_filler=dict(type='constant',value=0),
                                 exclude=dict(stage='deploy'))
    net.i2part_softmaxLoss=L.SoftmaxWithLoss(*[net.i2part_fc,net.label],
                                             exclude=dict(stage='deploy'))
    net.i2part_accuracy=L.Accuracy(*[net.i2part_fc,net.label],
                            include=dict(stage='val'))
    
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    with open(filename,'w') as f:
        f.write('%s\n'%net.to_proto())
        f.close()

def low_res_model_v2(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='trainval-data',
                    exclude=dict(stage='deploy'),
                    ntop=2)
    
    net.data_dropout=L.Dropout(net.data,
                               dropout_param={'dropout_ratio':0.5},
                               in_place=True,
                               include=dict(stage='train'))
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,64,lr_mults=(1,1))
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 32x32
    
    iconcat=addInception_v1_2(net.pool1,net, 'incep1', ksizes=[7,5,3], noutputs=[[32,64],[32,64],[32,64]], pads=[3,2,1], lr_mults=(1,1))
    
    net.bn1=L.BatchNorm(iconcat)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    ires=addInceptionRes2(net.pool2,net, 'incep2', ksizes=[9,7,5,3], noutputs=[[64,96],[64,96],[64,96],[64,96]], pads=[4,3,2,1], lr_mults=(1,1))
    
    net.bn2=L.BatchNorm(ires)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 8x8
    iconcat=addInception_v3_2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[384,512],[384,512]], pads=[2,1], lr_mults=(1,1))

    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':4,'stride':4})
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    net.i1part_pool=L.Pooling(net.bn1,
                              pooling_param={'pool':P.Pooling.MAX,'kernel_size':16,'stride':16},
                              exclude=dict(stage='deploy'))
    net.i1part_fc=L.InnerProduct(net.i1part_pool,
                                 param=[{'lr_mult':1,'decay_mult':2},
                                        {'lr_mult':1,'decay_mult':0}],
                                 weight_filler=dict(type='xavier'),
                                 bias_filler=dict(type='constant',value=0),
                                 exclude=dict(stage='deploy'))
    net.i1part_dropout=L.Dropout(net.i1part_fc,
                               dropout_param={'dropout_ratio':0.2},
                               in_place=True,
                               include=dict(stage='train'))
    net.i1part_softmaxLoss=L.SoftmaxWithLoss(*[net.i1part_fc,net.label],
                                             exclude=dict(stage='deploy'))
    net.i1part_accuracy=L.Accuracy(*[net.i1part_fc,net.label],
                            include=dict(stage='val'))
    
    net.i2part_pool=L.Pooling(net.bn2,
                              pooling_param={'pool':P.Pooling.MAX,'kernel_size':16,'stride':16},
                              exclude=dict(stage='deploy'))
    net.i2part_fc=L.InnerProduct(net.i2part_pool,
                                 param=[{'lr_mult':1,'decay_mult':2},
                                        {'lr_mult':1,'decay_mult':0}],
                                 weight_filler=dict(type='xavier'),
                                 bias_filler=dict(type='constant',value=0),
                                 exclude=dict(stage='deploy'))
    net.i2part_dropout=L.Dropout(net.i2part_fc,
                               dropout_param={'dropout_ratio':0.2},
                               in_place=True,
                               include=dict(stage='train'))
    net.i2part_softmaxLoss=L.SoftmaxWithLoss(*[net.i2part_fc,net.label],
                                             exclude=dict(stage='deploy'))
    net.i2part_accuracy=L.Accuracy(*[net.i2part_fc,net.label],
                            include=dict(stage='val'))
    
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    with open(filename,'w') as f:
        f.write('%s\n'%net.to_proto())
        f.close()

def low_res_model_v3(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='train-data',
                    exclude=dict(stage='deploy'),
                    transform_param={'crop_size':60},
                    ntop=2)
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,64,lr_mults=(1,1))
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 31x31
    
    iconcat=deeper_inception(net.pool1,net, 'incep1', ksizess=[[7,5,3],[7,5,3]], noutputss=[[[32,64],[32,64],[32,64]],[[64,128],[64,128],[64,128]]], padss=[[3,2,1],[3,2,1]],currentSize=31, lr_mults=(1,1))
    
    net.bn1=L.BatchNorm(iconcat)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    iconcat=addInceptionRes2(net.pool2,net, 'incep2', ksizes=[7,5,3], noutputs=[[128,192],[128,192],[128,192]], pads=[3,2,1], lr_mults=(1,1))
    
    net.bn2=L.BatchNorm(iconcat)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 9x9
    iconcat=addInceptionRes2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[256,512],[256,512]], pads=[2,1], lr_mults=(1,1))
    
    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':9,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([net.bn1,net.label], net, 'incep1', 31,feature_dim=512,loss_weight=0.7)
    addPartialRegression([net.bn2,net.label], net, 'incep2', 16,feature_dim=1024,loss_weight=0.7)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    exportModel(net, filename)

def low_res_model_v4(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='train-data',
                    exclude=dict(stage='deploy'),
                    transform_param={'crop_size':60},
                    ntop=2)
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,64,lr_mults=(1,1))
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 31x31
    
    iconcat=deeper_inception(net.pool1,net, 'incep1',
                              ksizess=[[7,5,3],[7,5,3],[7,5,3]], 
                              noutputss=[[[32,64],[32,64],[32,64]],[[64,96],[64,96],[64,96]],[[96,128],[96,128],[96,128]]], 
                              padss=[[3,2,1],[3,2,1],[3,2,1]], 
                              currentSize=31,
                              version='v4',
                              lr_mults=(1,1),
                              base_loss_weight=0.7)
    
    net.bn1=L.BatchNorm(iconcat)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    iconcat=addInceptionRes2(net.pool2,net, 'incep2', ksizes=[7,5,3], noutputs=[[128,192],[128,192],[128,192]], pads=[3,2,1], lr_mults=(1,1))
    
    net.bn2=L.BatchNorm(iconcat)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 9x9
    iconcat=addInceptionRes2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[256,512],[256,512]], pads=[2,1], lr_mults=(1,1))
    
    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':9,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([net.bn1,net.label], net, 'incep1', 31,loss_weight=0.7,feature_dim=512)
    addPartialRegression([net.bn2,net.label], net, 'incep2', 16,loss_weight=0.7,feature_dim=1024)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    exportModel(net, filename)

def low_res_model_v5(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='train-data',
                    exclude=dict(stage='deploy'),
                    transform_param={'crop_size':60},
                    ntop=2)
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,32,lr_mults=(1,1),pad=1)
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 31x31
    
    iconcat=deeper_inception(net.pool1,net, 'incep1',
                              ksizess=[[3,3,3],[5,5,5],[7,7,7]], 
                              noutputss=[[[32,64],[32,64],[32,64]],[[64,96],[64,96],[64,96]],[[96,128],[96,128],[96,128]]], 
                              padss=[[1,1,1],[2,2,2],[3,3,3]], 
                              currentSize=31,
                              version='v4',
                              lr_mults=(1,1),
                              base_loss_weight=0.1,
                              discount=1.1)
    
    net.bn1=L.BatchNorm(iconcat)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    iconcat=addInceptionRes2(net.pool2,net, 'incep2', ksizes=[7,5,3], noutputs=[[128,192],[128,192],[128,192]], pads=[3,2,1], lr_mults=(1,1))
    
    net.bn2=L.BatchNorm(iconcat)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 9x9
    iconcat=addInceptionRes2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[256,512],[256,512]], pads=[2,1], lr_mults=(1,1))
    
    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':9,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([net.bn1,net.label], net, 'incep1', 31,loss_weight=0.3,feature_dim=512)
    addPartialRegression([net.bn2,net.label], net, 'incep2', 16,dropout_ratio=0.5,loss_weight=0.7,feature_dim=1024)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    exportModel(net, filename)

def low_res_model_v6(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='train-data',
                    exclude=dict(stage='deploy'),
                    transform_param={'crop_size':60},
                    ntop=2)
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,32,lr_mults=(1,1),pad=1)
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 31x31
    
    iconcat=deeper_inception(net.pool1,net, 'incep1',
                              ksizess=[[3,3,3],[5,5,5],[7,7,7]], 
                              noutputss=[[[32,64],[32,64],[32,64]],[[64,96],[64,96],[64,96]],[[96,128],[96,128],[96,128]]], 
                              padss=[[1,1,1],[2,2,2],[3,3,3]], 
                              currentSize=31,
                              version='v4',
                              lr_mults=(1,1),
                              base_loss_weight=0.7,
                              discount=1,
                              scale_bn=True)
    
    #net.bn1=L.BatchNorm(iconcat)
    bn1=addBatchNorm(iconcat, net, 'bn1')
    net.pool2=L.Pooling(bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    iconcat=addInceptionRes2(net.pool2,net, 'incep2', ksizes=[7,5,3], noutputs=[[128,192],[128,192],[128,192]], pads=[3,2,1], lr_mults=(1,1))
    
    bn2=addBatchNorm(iconcat, net, 'bn2')
    
    net.pool3=L.Pooling(bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 9x9
    iconcat=addInceptionRes2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[256,512],[256,512]], pads=[2,1], lr_mults=(1,1))
    
    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.AVE,'kernel_size':9,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([bn1,net.label], net, 'incep1', 31,dropout_ratio=0.7,loss_weight=0.7,feature_dim=512)
    addPartialRegression([bn2,net.label], net, 'incep2', 16,dropout_ratio=0.7,loss_weight=0.7,feature_dim=1024)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    exportModel(net, filename)
    
def low_res_model_v7(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='train-data',
                    exclude=dict(stage='deploy'),
                    transform_param={'crop_size':60},
                    ntop=2)
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,32,lr_mults=(1,1),pad=1)
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 31x31
    
    iconcat=deeper_inception(net.pool1,net, 'incep1',
                              ksizess=[[3,3,3],[5,5,5],[7,7,7]], 
                              noutputss=[[[32,64],[32,64],[32,64]],[[64,96],[64,96],[64,96]],[[96,128],[96,128],[96,128]]], 
                              padss=[[1,1,1],[2,2,2],[3,3,3]], 
                              currentSize=31,
                              version='v4',
                              lr_mults=(1,1),
                              base_loss_weight=0.1,
                              discount=1.1,
                              scale_bn=True)
    
    #net.bn1=L.BatchNorm(iconcat)
    bn1=addBatchNorm(iconcat, net, 'bn1')
    net.pool2=L.Pooling(bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    iconcat=addInceptionRes2(net.pool2,net, 'incep2', ksizes=[7,5,3], noutputs=[[128,192],[128,192],[128,192]], pads=[3,2,1], lr_mults=(1,1))
    
    bn2=addBatchNorm(iconcat, net, 'bn2')
    
    net.pool3=L.Pooling(bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 9x9
    iconcat=addInceptionRes2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[256,512],[256,512]], pads=[2,1], lr_mults=(1,1))
    
    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':9,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    exportModel(net, filename)

def low_res_model_v8(filename):
    # for 64x64
    net=caffe.NetSpec()
    #net.name='LRNet'
    net.data,net.label=L.Data(name='train-data',
                    exclude=dict(stage='deploy'),
                    transform_param={'crop_size':60},
                    ntop=2)
    
    net.conv1,net.relu1=addConvolution(net.data,'1',3,32,lr_mults=(1,1),pad=1)
    net.pool1=L.Pooling(net.conv1,name='pool1',
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 31x31
    
    iconcat=deeper_inception(net.pool1,net, 'incep1',
                              ksizess=[[3,3,3],[5,5,5],[7,7,7]], 
                              noutputss=[[[32,64],[32,64],[32,64]],[[64,96],[64,96],[64,96]],[[96,128],[96,128],[96,128]]], 
                              padss=[[1,1,1],[2,2,2],[3,3,3]], 
                              currentSize=31,
                              version='v4',
                              lr_mults=(1,1),
                              base_loss_weight=0.7,
                              discount=1,
                              scale_bn=True)
    
    #net.bn1=L.BatchNorm(iconcat)
    bn1=addBatchNorm(iconcat, net, 'bn1')
    net.pool2=L.Pooling(bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 16x16
    
    iconcat=addInceptionRes2(net.pool2,net, 'incep2', ksizes=[7,5,3], noutputs=[[128,192],[128,192],[128,192]], pads=[3,2,1], lr_mults=(1,1))
    
    bn2=addBatchNorm(iconcat, net, 'bn2')
    
    net.pool3=L.Pooling(bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    
    # 9x9
    iconcat=addInceptionRes2(net.pool3,net, 'incep3', ksizes=[5,3], noutputs=[[256,512],[256,512]], pads=[2,1], lr_mults=(1,1))
    
    net.pool4=L.Pooling(iconcat,
                        pooling_param={'pool':P.Pooling.AVE,'kernel_size':9,'stride':1})
    net.pool4=L.Dropout(net.pool4,
                        dropout_param={'dropout_ratio':0.5},
                        in_place=True)
    
    net.classifier=L.InnerProduct(net.pool4,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([bn1,net.label], net, 'incep1', 31,dropout_ratio=0.7,loss_weight=0.7,feature_dim=512)
    addPartialRegression([bn2,net.label], net, 'incep2', 16,dropout_ratio=0.7,loss_weight=0.7,feature_dim=1024)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    exportModel(net, filename)

def mid_res_model(filename):
    # Input size is 227
    net=caffe.NetSpec()
    net.data,net.label=L.Data(name='trainval-data',
                    exclude=dict(stage='deploy'),
                    ntop=2)
    net.conv1,net.relu1=addConvolution(net.data,'1',3,32,lr_mults=(1,1))
    net.pool1=L.Pooling(net.conv1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #114
    net.conv2,net.relu2=addConvolution(net.pool1, '2', 3, 64, lr_mults=(1,1))
    net.bn1=L.BatchNorm(net.conv2)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #58
    ires=addInceptionRes2(net.pool2,net, 'incep1', ksizes=[7,5,3], noutputs=[[64,96],[64,96],[64,96]], pads=[3,2,1], lr_mults=(1,1,1))
    
    net.bn2=L.BatchNorm(ires)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #30
    ires=addInceptionRes2(net.pool3,net, 'incep2', ksizes=[7,5,3], noutputs=[[96,128],[96,128],[96,128]], pads=[3,2,1], lr_mults=(1,1,1))
    
    net.bn3=L.BatchNorm(ires)
    net.pool4=L.Pooling(net.bn3,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #16
    ires=addInceptionRes2(net.pool4,net, 'incep3', ksizes=[7,5,3], noutputs=[[128,192],[128,192],[128,192]], pads=[3,2,1], lr_mults=(1,1,1))
    
    
    net.bn4=L.BatchNorm(ires)
    net.pool5=L.Pooling(net.bn4,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':16,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool5,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([net.bn2,net.label],net,'incep1',58,loss_weight=0.3,feature_dim=512)
    addPartialRegression([net.bn3,net.label],net,'incep2',30,loss_weight=0.7,feature_dim=1024)
    
    net.final_relu=L.ReLU(net.classifier,in_place=True)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    
    exportModel(net, filename)
    
def mid_res_model_v2(filename):
    # Input size is 227
    net=caffe.NetSpec()
    net.data,net.label=L.Data(name='trainval-data',
                    exclude=dict(stage='deploy'),
                    ntop=2)
    net.conv1,net.relu1=addConvolution(net.data,'1',3,32,lr_mults=(1,1))
    net.pool1=L.Pooling(net.conv1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #114
    net.conv2,net.relu2=addConvolution(net.pool1, '2', 3, 64, lr_mults=(1,1))
    net.bn1=L.BatchNorm(net.conv2)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #58
    ires=deeper_inception(net.pool2,net, 'incep1', 
                          ksizess=[[7,5,3],[7,5,3],[7,5,3]], 
                          noutputss=[[[64,96],[64,96],[64,96]],[[96,128],[96,128],[96,128]],[[128,160],[128,160],[128,160]]], 
                          padss=[[3,2,1],[3,2,1],[3,2,1]],
                          currentSize=58
                          , lr_mults=(1,1,1))
    
    net.bn2=L.BatchNorm(ires)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #30
    ires=addInceptionRes2(net.pool3,net, 'incep2', ksizes=[7,5,3], noutputs=[[160,192],[160,192],[160,192]], pads=[3,2,1], lr_mults=(1,1,1))
    
    net.bn3=L.BatchNorm(ires)
    net.pool4=L.Pooling(net.bn3,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #16
    ires=addInceptionRes2(net.pool4,net, 'incep3', ksizes=[7,5,3], noutputs=[[192,224],[192,224],[192,224]], pads=[3,2,1], lr_mults=(1,1,1))
    
    
    net.bn4=L.BatchNorm(ires)
    net.pool5=L.Pooling(net.bn4,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':16,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool5,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([net.bn2,net.label],net,'incep1',58,loss_weight=0.7,feature_dim=512)
    addPartialRegression([net.bn3,net.label],net,'incep2',30,loss_weight=0.7,feature_dim=1024)
    
    net.final_relu=L.ReLU(net.classifier,in_place=True)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    
    exportModel(net, filename)
    
def mid_res_model_v3(filename):
    # Input size is 227
    net=caffe.NetSpec()
    net.data,net.label=L.Data(name='trainval-data',
                    exclude=dict(stage='deploy'),
                    ntop=2)
    net.conv1,net.relu1=addConvolution(net.data,'1',3,32,lr_mults=(1,1))
    net.pool1=L.Pooling(net.conv1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #114
    net.conv2,net.relu2=addConvolution(net.pool1, '2', 3, 64, lr_mults=(1,1))
    net.bn1=L.BatchNorm(net.conv2)
    net.pool2=L.Pooling(net.bn1,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #58
    ires=deeper_inception(net.pool2,net, 'incep1', 
                          ksizess=[[7,7,3],[5,5,3]], 
                          noutputss=[[[64,96],[64,96],[64,96]],[[96,128],[96,128],[96,128]]], 
                          padss=[[3,3,1],[2,2,1],[1,1,1]],
                          currentSize=58,
                          version='v4'
                          , lr_mults=(1,1))
    
    net.bn2=L.BatchNorm(ires)
    net.pool3=L.Pooling(net.bn2,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #30
    ires=addInceptionRes2(net.pool3,net, 'incep2', ksizes=[7,5,3], noutputs=[[128,160],[128,160],[128,160]], pads=[3,2,1], lr_mults=(1,1,1))
    
    net.bn3=L.BatchNorm(ires)
    net.pool4=L.Pooling(net.bn3,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':3,'stride':2,'pad':1})
    #16
    ires=addInceptionRes2(net.pool4,net, 'incep3', ksizes=[5,5,3], noutputs=[[160,192],[160,192],[160,192]], pads=[2,2,1], lr_mults=(1,1,1))
    #addResidual(net.pool4,[ires],net,'ext_residual1')
    
    net.bn4=L.BatchNorm(ires)
    net.pool5=L.Pooling(net.bn4,
                        pooling_param={'pool':P.Pooling.MAX,'kernel_size':16,'stride':1})
    
    net.classifier=L.InnerProduct(net.pool5,
                                  param=[{'lr_mult':1,'decay_mult':2},
                                         {'lr_mult':1,'decay_mult':0}],
                                  weight_filler=dict(type='xavier'),
                                  bias_filler=dict(type='constant',value=0))
    
    addPartialRegression([net.bn2,net.label],net,'incep1',58,loss_weight=0.5,feature_dim=512)
    addPartialRegression([net.bn3,net.label],net,'incep2',30,loss_weight=0.7,feature_dim=1024)
    
    net.final_relu=L.ReLU(net.classifier,in_place=True)
    
    net.softmaxLoss=L.SoftmaxWithLoss(*[net.classifier,net.label],
                                  exclude=dict(stage='deploy'))
    net.softmax=L.Softmax(net.classifier,
                          include=dict(stage='deploy'))
    net.accuracy=L.Accuracy(*[net.classifier,net.label],
                            include=dict(stage='val'))
    
    
    exportModel(net, filename)
    
mid_res_model('mid_res_model.prototxt')
mid_res_model_v2('mid_res_model_v2.prototxt')
mid_res_model_v3('mid_res_model_v3.prototxt')
low_res_model('low_res_model.prototxt')
low_res_model_v2('low_res_model_v2.prototxt')
low_res_model_v3('low_res_model_v3.prototxt')
low_res_model_v4('low_res_model_v4.prototxt')
low_res_model_v5('low_res_model_v5.prototxt')
low_res_model_v6('low_res_model_v6.prototxt')
low_res_model_v7('low_res_model_v7.prototxt')
low_res_model_v8('low_res_model_v8.prototxt')