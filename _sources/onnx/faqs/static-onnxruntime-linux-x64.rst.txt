undefined reference to `std::__throw_bad_array_new_length()'
============================================================

If you get the following error when using ``-DBUILD_SHARED_LIBS=OFF``
on Linux with x64::

  /opt/rh/devtoolset-10/root/usr/libexec/gcc/x86_64-redhat-linux/10/ld: ../../_deps/onnxruntime-src/lib/libonnxruntime.a(implementation.cc.o): in function `std::_Hashtable<std::string, std::pair<std::string const, onnx::AttributeProto const*>, std::allocator<std::pair<std::string const, onnx::AttributeProto const*> >, std::__detail::_Select1st, std::equal_to<std::string>, std::hash<std::string>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<true, false, true> >::_M_rehash(unsigned long, unsigned long const&)':
  implementation.cc:(.text._ZNSt10_HashtableISsSt4pairIKSsPKN4onnx14AttributeProtoEESaIS6_ENSt8__detail10_Select1stESt8equal_toISsESt4hashISsENS8_18_Mod_range_hashingENS8_20_Default_ranged_hashENS8_20_Prime_rehash_policyENS8_17_Hashtable_traitsILb1ELb0ELb1EEEE9_M_rehashEmRKm[_ZNSt10_HashtableISsSt4pairIKSsPKN4onnx14AttributeProtoEESaIS6_ENSt8__detail10_Select1stESt8equal_toISsESt4hashISsENS8_18_Mod_range_hashingENS8_20_Default_ranged_hashENS8_20_Prime_rehash_policyENS8_17_Hashtable_traitsILb1ELb0ELb1EEEE9_M_rehashEmRKm]+0x10f): undefined reference to `std::__throw_bad_array_new_length()'

Please either switch to a new GCC, e.g., ``GCC >= 11`` or use ``-DBUILD_SHARED_LIBS=ON`` when
building `sherpa-onnx`_.

.. hint::

   Remember to delete the build directory of `sherpa-onnx`_ before you retry.
