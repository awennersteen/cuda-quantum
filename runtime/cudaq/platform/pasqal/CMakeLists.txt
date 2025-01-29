# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(cudaq-rest-qpu PRIVATE PasqalServerHelper.cpp)

add_library(cudaq-serverhelper-pasqal SHARED PasqalServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-pasqal
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-pasqal DESTINATION lib)
