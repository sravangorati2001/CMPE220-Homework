//
//  MetalUtils.h
//  CMPE220_metal
//
//  Created by Sravan Kumar Gorati on 5/10/24.
//

#ifndef METALUTILS_H
#define METALUTILS_H

#include <vector>
#import <Metal/Metal.h>

// Function to create Metal buffer
id<MTLBuffer> createBuffer(id<MTLDevice> device, const std::vector<float>& array);

#endif // METALUTILS_H

