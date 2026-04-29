import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

/// Dart FFI bindings for the Gollek Local SDK.
class GollekFfiBindings {
  late final DynamicLibrary _lib;

  // Lifecycle
  late final int Function(Pointer<NativeType>) _createClient;
  late final int Function(Pointer<NativeType>, int) _destroyClient;
  late final void Function(Pointer<NativeType>) _shutdownRuntime;

  // Error Handling
  late final Pointer<Utf8> Function(Pointer<NativeType>) _lastError;
  late final void Function(Pointer<NativeType>) _clearLastError;
  late final void Function(Pointer<NativeType>, Pointer<Utf8>) _freeString;

  // Inference (JSON based)
  late final Pointer<Utf8> Function(Pointer<NativeType>, int, Pointer<Utf8>) _createCompletion;
  late final Pointer<Utf8> Function(Pointer<NativeType>, int, Pointer<Utf8>) _streamCompletion;
  
  // Model Management
  late final Pointer<Utf8> Function(Pointer<NativeType>, int) _listModels;
  late final Pointer<Utf8> Function(Pointer<NativeType>, int, Pointer<Utf8>) _getModelInfo;

  GollekFfiBindings({String? libraryPath}) {
    final path = libraryPath ?? _resolveLibraryPath();
    _lib = DynamicLibrary.open(path);
    _initializeFunctions();
  }

  void _initializeFunctions() {
    _createClient = _lib.lookupFunction<Int64 Function(Pointer<NativeType>), int Function(Pointer<NativeType>)>('golek_client_create');
    _destroyClient = _lib.lookupFunction<Int32 Function(Pointer<NativeType>, Int64), int Function(Pointer<NativeType>, int)>('golek_client_destroy');
    _shutdownRuntime = _lib.lookupFunction<Void Function(Pointer<NativeType>), void Function(Pointer<NativeType>)>('golek_client_shutdown_runtime');

    _lastError = _lib.lookupFunction<Pointer<Utf8> Function(Pointer<NativeType>), Pointer<Utf8> Function(Pointer<NativeType>)>('golek_last_error');
    _clearLastError = _lib.lookupFunction<Void Function(Pointer<NativeType>), void Function(Pointer<NativeType>)>('golek_clear_last_error');
    _freeString = _lib.lookupFunction<Void Function(Pointer<NativeType>, Pointer<Utf8>), void Function(Pointer<NativeType>, Pointer<Utf8>)>('golek_string_free');

    _createCompletion = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<NativeType>, Int64, Pointer<Utf8>),
        Pointer<Utf8> Function(Pointer<NativeType>, int, Pointer<Utf8>)>('golek_create_completion_json');
    
    _streamCompletion = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<NativeType>, Int64, Pointer<Utf8>),
        Pointer<Utf8> Function(Pointer<NativeType>, int, Pointer<Utf8>)>('golek_stream_completion_json');

    _listModels = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<NativeType>, Int64),
        Pointer<Utf8> Function(Pointer<NativeType>, int)>('golek_list_models_json');

    _getModelInfo = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<NativeType>, Int64, Pointer<Utf8>),
        Pointer<Utf8> Function(Pointer<NativeType>, int, Pointer<Utf8>)>('golek_get_model_info_json');
  }

  String _resolveLibraryPath() {
    if (Platform.isMacOS) return 'libgollek_sdk_local.dylib';
    if (Platform.isLinux || Platform.isAndroid) return 'libgollek_sdk_local.so';
    if (Platform.isWindows) return 'gollek_sdk_local.dll';
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  // --- Helper to handle GraalVM IsolateThread ---
  // In a simple single-threaded app, we can just pass nullptr for the thread.
  // GraalVM will automatically attach/create a thread if needed.
  
  String _consumeString(Pointer<Utf8> ptr) {
    if (ptr == nullptr) return "";
    final str = ptr.toDartString();
    _freeString(nullptr, ptr);
    return str;
  }

  // --- Public API ---

  int createClient() {
    final handle = _createClient(nullptr);
    if (handle == 0) {
      throw Exception('Failed to create Gollek client: ${_consumeString(_lastError(nullptr))}');
    }
    return handle;
  }

  void destroyClient(int handle) {
    if (_destroyClient(nullptr, handle) == 0) {
      throw Exception('Failed to destroy Gollek client: ${_consumeString(_lastError(nullptr))}');
    }
  }

  String createCompletion(int clientHandle, String requestJson) {
    final jsonPtr = requestJson.toNativeUtf8();
    try {
      final resPtr = _createCompletion(nullptr, clientHandle, jsonPtr);
      if (resPtr == nullptr) {
        throw Exception('Inference failed: ${_consumeString(_lastError(nullptr))}');
      }
      return _consumeString(resPtr);
    } finally {
      calloc.free(jsonPtr);
    }
  }

  String listModels(int clientHandle) {
    final resPtr = _listModels(nullptr, clientHandle);
    return _consumeString(resPtr);
  }
}
