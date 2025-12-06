import { handleRequest } from "./index";

export const onRequest = async (context: { request: Request }): Promise<Response> => {
  return handleRequest(context.request);
};
